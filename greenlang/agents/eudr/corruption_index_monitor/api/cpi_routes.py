# -*- coding: utf-8 -*-
"""
CPI Monitoring Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for Transparency International Corruption Perceptions Index (CPI)
score monitoring including single country score retrieval, historical score
tracking, global/regional rankings, regional analysis, batch multi-country
queries, and summary statistics.

Endpoints:
    GET  /cpi/{country_code}/score    - Get current CPI score for a country
    GET  /cpi/{country_code}/history  - Get CPI score history with year range
    GET  /cpi/rankings                - Global/regional CPI rankings
    GET  /cpi/regional/{region}       - Regional CPI analysis
    POST /cpi/batch-query             - Batch query multiple countries
    GET  /cpi/summary                 - CPI summary statistics

CPI Scale: 0-100 (0=most corrupt, 100=cleanest)
Data Source: Transparency International
Coverage: 180+ countries

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, CPI Monitor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.corruption_index_monitor.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_cpi_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
    validate_country_code,
    validate_year_range,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    CPIBatchRequest,
    CPIBatchResponse,
    CPIBatchResultEntry,
    CPIHistoryEntry,
    CPIHistoryResponse,
    CPIRankingEntry,
    CPIRankingsResponse,
    CPIRegionalResponse,
    CPIRegionalStats,
    CPIScoreEntry,
    CPIScoreResponse,
    CPISummaryResponse,
    CountryClassificationEnum,
    DataSourceEnum,
    ErrorResponse as SchemaErrorResponse,
    HealthResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RegionEnum,
    RiskLevelEnum,
    TrendDirectionEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cpi", tags=["CPI Monitoring"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_cpi_risk(score: Decimal) -> RiskLevelEnum:
    """Classify CPI score into a risk level.

    Args:
        score: CPI score on 0-100 scale.

    Returns:
        Risk level classification.
    """
    if score <= Decimal("30"):
        return RiskLevelEnum.CRITICAL
    elif score <= Decimal("50"):
        return RiskLevelEnum.HIGH
    elif score <= Decimal("70"):
        return RiskLevelEnum.MODERATE
    return RiskLevelEnum.LOW


def _classify_eudr(score: Decimal) -> CountryClassificationEnum:
    """Map CPI score to EUDR Article 29 country classification.

    Args:
        score: CPI score on 0-100 scale.

    Returns:
        EUDR country classification.
    """
    if score >= Decimal("60"):
        return CountryClassificationEnum.LOW_RISK
    elif score <= Decimal("30"):
        return CountryClassificationEnum.HIGH_RISK
    return CountryClassificationEnum.STANDARD_RISK


# ---------------------------------------------------------------------------
# GET /cpi/{country_code}/score
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/score",
    response_model=CPIScoreResponse,
    summary="Get CPI score for a country",
    description=(
        "Retrieve the current Corruption Perceptions Index (CPI) score for a "
        "specific country. Returns the score on a 0-100 scale (0=most corrupt, "
        "100=cleanest), global ranking position, percentile, and EUDR Article 29 "
        "country classification. Data sourced from Transparency International."
    ),
    responses={
        200: {"description": "CPI score retrieved successfully"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def get_cpi_score(
    country_code: str,
    request: Request,
    year: Optional[int] = Query(
        None,
        ge=1995,
        le=2030,
        description="CPI year (default: latest available)",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CPIScoreResponse:
    """Get the current CPI score for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        year: Optional specific year (default: latest).
        user: Authenticated user with cpi:read permission.

    Returns:
        CPIScoreResponse with score, risk level, and EUDR classification.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_cpi_engine()
        result = engine.get_score(normalized_code, year=year)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CPI score not found for country {normalized_code}",
            )

        score_val = Decimal(str(result.get("score", 0)))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"{normalized_code}:{result_year}", str(score_val)
        )

        score_entry = CPIScoreEntry(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            year=result_year,
            score=score_val,
            rank=result.get("rank"),
            percentile=Decimal(str(result.get("percentile", 0))) if result.get("percentile") else None,
            region=result.get("region"),
            year_over_year_change=Decimal(str(result.get("yoy_change", 0))) if result.get("yoy_change") else None,
            data_source=DataSourceEnum.TRANSPARENCY_INTERNATIONAL,
        )

        risk_level = _classify_cpi_risk(score_val)
        classification = _classify_eudr(score_val)

        logger.info(
            "CPI score retrieved: country=%s year=%d score=%s risk=%s operator=%s",
            normalized_code,
            result_year,
            score_val,
            risk_level.value,
            user.operator_id or user.user_id,
        )

        return CPIScoreResponse(
            score=score_entry,
            risk_level=risk_level,
            eudr_classification=classification,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI score retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI score retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /cpi/{country_code}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/history",
    response_model=CPIHistoryResponse,
    summary="Get CPI score history for a country",
    description=(
        "Retrieve historical CPI scores for a country over a specified year "
        "range. Includes year-over-year changes, average score, and trend "
        "direction analysis. Minimum 2 data points required for trend detection."
    ),
    responses={
        200: {"description": "CPI history retrieved successfully"},
        400: {"model": SchemaErrorResponse, "description": "Invalid parameters"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_cpi_history(
    country_code: str,
    request: Request,
    year_range: Dict[str, Optional[int]] = Depends(validate_year_range),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CPIHistoryResponse:
    """Get CPI score history for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        year_range: Optional start/end year range.
        user: Authenticated user with cpi:read permission.

    Returns:
        CPIHistoryResponse with historical scores and trend analysis.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_cpi_engine()
        result = engine.get_history(
            normalized_code,
            start_year=year_range.get("start_year"),
            end_year=year_range.get("end_year"),
        )

        if result is None or not result.get("history"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CPI history not found for country {normalized_code}",
            )

        history_entries = []
        prev_score = None
        for entry in result.get("history", []):
            score = Decimal(str(entry.get("score", 0)))
            change = None
            if prev_score is not None:
                change = score - prev_score
            history_entries.append(
                CPIHistoryEntry(
                    year=entry.get("year", 2024),
                    score=score,
                    rank=entry.get("rank"),
                    change_from_prior=change,
                )
            )
            prev_score = score

        scores = [e.score for e in history_entries]
        avg_score = sum(scores) / len(scores) if scores else Decimal("0")
        total_change = scores[-1] - scores[0] if len(scores) >= 2 else Decimal("0")

        if total_change > Decimal("2"):
            trend = TrendDirectionEnum.IMPROVING
        elif total_change < Decimal("-2"):
            trend = TrendDirectionEnum.DETERIORATING
        else:
            trend = TrendDirectionEnum.STABLE

        period_start = history_entries[0].year if history_entries else 2020
        period_end = history_entries[-1].year if history_entries else 2024

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"{normalized_code}:{period_start}-{period_end}",
            str(avg_score),
        )

        logger.info(
            "CPI history retrieved: country=%s period=%d-%d entries=%d operator=%s",
            normalized_code,
            period_start,
            period_end,
            len(history_entries),
            user.operator_id or user.user_id,
        )

        return CPIHistoryResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            history=history_entries,
            period_start=period_start,
            period_end=period_end,
            average_score=avg_score.quantize(Decimal("0.01")),
            trend_direction=trend,
            total_change=total_change,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI history retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI history retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /cpi/rankings
# ---------------------------------------------------------------------------


@router.get(
    "/rankings",
    response_model=CPIRankingsResponse,
    summary="Get global/regional CPI rankings",
    description=(
        "Retrieve CPI rankings sorted by score. Optionally filter by region. "
        "Returns ranked list with risk classifications and pagination support."
    ),
    responses={
        200: {"description": "CPI rankings retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def get_cpi_rankings(
    request: Request,
    year: Optional[int] = Query(None, ge=1995, le=2030, description="Rankings year"),
    region: Optional[RegionEnum] = Query(None, description="Region filter"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CPIRankingsResponse:
    """Get CPI rankings globally or filtered by region.

    Args:
        year: Rankings year (default: latest).
        region: Optional region filter.
        pagination: Pagination parameters.
        user: Authenticated user with cpi:read permission.

    Returns:
        CPIRankingsResponse with ranked countries.
    """
    start = time.monotonic()

    try:
        engine = get_cpi_engine()
        result = engine.get_rankings(
            year=year,
            region=region.value if region else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        rankings = []
        for entry in result.get("rankings", []):
            score = Decimal(str(entry.get("score", 0)))
            rankings.append(
                CPIRankingEntry(
                    rank=entry.get("rank", 0),
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    score=score,
                    region=entry.get("region"),
                    year_over_year_change=Decimal(str(entry.get("yoy_change", 0))) if entry.get("yoy_change") else None,
                    risk_level=_classify_cpi_risk(score),
                )
            )

        total = result.get("total", len(rankings))
        global_avg = Decimal(str(result.get("global_average", 43)))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"rankings:{result_year}:{region}", str(total)
        )

        logger.info(
            "CPI rankings retrieved: year=%d region=%s total=%d operator=%s",
            result_year,
            region.value if region else "global",
            total,
            user.operator_id or user.user_id,
        )

        return CPIRankingsResponse(
            year=result_year,
            region=region,
            rankings=rankings,
            total_countries=total,
            global_average=global_avg,
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
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI rankings retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI rankings retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /cpi/regional/{region}
# ---------------------------------------------------------------------------


@router.get(
    "/regional/{region}",
    response_model=CPIRegionalResponse,
    summary="Get regional CPI analysis",
    description=(
        "Retrieve detailed CPI analysis for a specific region including "
        "statistical summary (mean, median, std dev), top and bottom performers, "
        "and high/low risk country counts."
    ),
    responses={
        200: {"description": "Regional analysis retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid region"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_cpi_regional(
    region: RegionEnum,
    request: Request,
    year: Optional[int] = Query(None, ge=1995, le=2030, description="Analysis year"),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CPIRegionalResponse:
    """Get regional CPI analysis with statistics and top/bottom performers.

    Args:
        region: Region to analyze.
        year: Analysis year (default: latest).
        user: Authenticated user with cpi:read permission.

    Returns:
        CPIRegionalResponse with regional statistics.
    """
    start = time.monotonic()

    try:
        engine = get_cpi_engine()
        result = engine.get_regional_analysis(
            region=region.value,
            year=year,
        )

        stats_data = result.get("stats", {})
        stats = CPIRegionalStats(
            region=region,
            country_count=stats_data.get("country_count", 0),
            average_score=Decimal(str(stats_data.get("average", 43))),
            median_score=Decimal(str(stats_data.get("median", 40))),
            min_score=Decimal(str(stats_data.get("min", 10))),
            max_score=Decimal(str(stats_data.get("max", 85))),
            std_deviation=Decimal(str(stats_data.get("std_dev", 15))),
            high_risk_count=stats_data.get("high_risk_count", 0),
            low_risk_count=stats_data.get("low_risk_count", 0),
        )

        top_performers = []
        for entry in result.get("top_performers", []):
            score = Decimal(str(entry.get("score", 0)))
            top_performers.append(
                CPIRankingEntry(
                    rank=entry.get("rank", 0),
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    score=score,
                    region=region,
                    risk_level=_classify_cpi_risk(score),
                )
            )

        bottom_performers = []
        for entry in result.get("bottom_performers", []):
            score = Decimal(str(entry.get("score", 0)))
            bottom_performers.append(
                CPIRankingEntry(
                    rank=entry.get("rank", 0),
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    score=score,
                    region=region,
                    risk_level=_classify_cpi_risk(score),
                )
            )

        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"regional:{region.value}:{result_year}",
            str(stats.average_score),
        )

        logger.info(
            "CPI regional analysis: region=%s year=%d countries=%d operator=%s",
            region.value,
            result_year,
            stats.country_count,
            user.operator_id or user.user_id,
        )

        return CPIRegionalResponse(
            year=result_year,
            region=region,
            stats=stats,
            top_performers=top_performers,
            bottom_performers=bottom_performers,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI regional analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI regional analysis failed",
        )


# ---------------------------------------------------------------------------
# POST /cpi/batch-query
# ---------------------------------------------------------------------------


@router.post(
    "/batch-query",
    response_model=CPIBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch query CPI scores for multiple countries",
    description=(
        "Query CPI scores for up to 100 countries in a single request. "
        "Each country is queried independently; partial failures return "
        "per-country error messages without failing the entire batch."
    ),
    responses={
        200: {"description": "Batch query completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_query_cpi(
    request: Request,
    body: CPIBatchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CPIBatchResponse:
    """Batch query CPI scores for multiple countries.

    Args:
        body: Batch request with country codes and optional year.
        user: Authenticated user with cpi:read permission.

    Returns:
        CPIBatchResponse with per-country results.
    """
    start = time.monotonic()

    try:
        engine = get_cpi_engine()
        results: List[CPIBatchResultEntry] = []
        succeeded = 0
        failed = 0

        for country_entry in body.countries:
            try:
                query_year = country_entry.year or body.year
                result = engine.get_score(
                    country_entry.country_code,
                    year=query_year,
                )

                if result is not None:
                    score_val = Decimal(str(result.get("score", 0)))
                    score_entry = CPIScoreEntry(
                        country_code=country_entry.country_code,
                        country_name=result.get("country_name", ""),
                        year=result.get("year", query_year or 2024),
                        score=score_val,
                        rank=result.get("rank"),
                        percentile=Decimal(str(result.get("percentile", 0))) if result.get("percentile") else None,
                        region=result.get("region"),
                        data_source=DataSourceEnum.TRANSPARENCY_INTERNATIONAL,
                    )
                    results.append(
                        CPIBatchResultEntry(
                            country_code=country_entry.country_code,
                            country_name=result.get("country_name", ""),
                            score=score_entry,
                            risk_level=_classify_cpi_risk(score_val),
                        )
                    )
                    succeeded += 1
                else:
                    results.append(
                        CPIBatchResultEntry(
                            country_code=country_entry.country_code,
                            error=f"No CPI data available for {country_entry.country_code}",
                        )
                    )
                    failed += 1

            except Exception as exc:
                results.append(
                    CPIBatchResultEntry(
                        country_code=country_entry.country_code,
                        error=f"Query failed: {str(exc)}",
                    )
                )
                failed += 1

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch:{len(body.countries)}", f"{succeeded}/{failed}"
        )

        logger.info(
            "CPI batch query: total=%d succeeded=%d failed=%d operator=%s",
            len(body.countries),
            succeeded,
            failed,
            user.operator_id or user.user_id,
        )

        return CPIBatchResponse(
            results=results,
            total_queried=len(body.countries),
            total_succeeded=succeeded,
            total_failed=failed,
            processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI batch query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI batch query failed",
        )


# ---------------------------------------------------------------------------
# GET /cpi/summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=CPISummaryResponse,
    summary="Get CPI summary statistics",
    description=(
        "Retrieve global CPI summary statistics including country counts by "
        "risk level, regional averages, year-over-year trends, and counts of "
        "improving/deteriorating countries."
    ),
    responses={
        200: {"description": "CPI summary retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_cpi_summary(
    request: Request,
    year: Optional[int] = Query(None, ge=1995, le=2030, description="Summary year"),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:cpi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CPISummaryResponse:
    """Get CPI summary statistics for all monitored countries.

    Args:
        year: Summary year (default: latest).
        user: Authenticated user with cpi:read permission.

    Returns:
        CPISummaryResponse with global and regional statistics.
    """
    start = time.monotonic()

    try:
        engine = get_cpi_engine()
        result = engine.get_summary(year=year)

        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"summary:{result_year}",
            str(result.get("total_countries", 0)),
        )

        regional_avgs = {}
        for region, avg in result.get("regional_averages", {}).items():
            regional_avgs[region] = Decimal(str(avg))

        logger.info(
            "CPI summary retrieved: year=%d countries=%d operator=%s",
            result_year,
            result.get("total_countries", 0),
            user.operator_id or user.user_id,
        )

        return CPISummaryResponse(
            year=result_year,
            total_countries=result.get("total_countries", 0),
            global_average=Decimal(str(result.get("global_average", 43))),
            global_median=Decimal(str(result.get("global_median", 40))),
            high_risk_countries=result.get("high_risk_countries", 0),
            moderate_risk_countries=result.get("moderate_risk_countries", 0),
            low_risk_countries=result.get("low_risk_countries", 0),
            regional_averages=regional_avgs,
            year_over_year_change=Decimal(str(result.get("yoy_change", 0))),
            improving_countries=result.get("improving_countries", 0),
            deteriorating_countries=result.get("deteriorating_countries", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CPI summary retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CPI summary retrieval failed",
        )
