# -*- coding: utf-8 -*-
"""
Trend Analysis Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for corruption index trend analysis including temporal trend
detection, country trajectory tracking, future prediction with confidence
intervals, and identification of improving/deteriorating countries.

Endpoints:
    POST /trends/analyze               - Analyze corruption trends
    GET  /trends/{country_code}/trajectory - Country trajectory
    POST /trends/prediction            - Future prediction with CI
    GET  /trends/improving             - Countries with improving trends
    GET  /trends/deteriorating         - Countries with deteriorating trends

Analysis requires minimum 5-year data windows, uses linear regression with
R-squared goodness of fit, and supports 3-year prediction horizons.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Trend Analysis Engine
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
    get_trend_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    DeterioratingCountriesResponse,
    ErrorResponse as SchemaErrorResponse,
    ImprovingCountriesResponse,
    MetadataSchema,
    PaginatedMeta,
    PredictionConfidenceEnum,
    PredictionRequest,
    PredictionResponse,
    ProvenanceInfo,
    RegionEnum,
    RiskLevelEnum,
    TrajectoryResponse,
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    TrendCountryEntry,
    TrendDataPoint,
    TrendDirectionEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trends", tags=["Trend Analysis"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /trends/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=TrendAnalysisResponse,
    summary="Analyze corruption trends for a country",
    description=(
        "Perform temporal trend analysis on corruption indices for a country. "
        "Uses linear regression to determine trend direction, slope, R-squared "
        "goodness of fit, and detects trend reversals. Requires minimum 5-year "
        "data window for valid trend analysis."
    ),
    responses={
        200: {"description": "Trend analysis completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_trends(
    request: Request,
    body: TrendAnalysisRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:trends:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TrendAnalysisResponse:
    """Analyze corruption trends for a specific country.

    Args:
        body: Trend analysis request with country, index type, and year range.
        user: Authenticated user with trends:create permission.

    Returns:
        TrendAnalysisResponse with regression results and trend detection.
    """
    start = time.monotonic()

    try:
        engine = get_trend_engine()
        result = engine.analyze_trend(
            country_code=body.country_code,
            index_type=body.index_type,
            start_year=body.start_year,
            end_year=body.end_year,
            wgi_dimension=body.wgi_dimension.value if body.wgi_dimension else None,
        )

        data_points = []
        for dp in result.get("data_points", []):
            data_points.append(
                TrendDataPoint(
                    year=dp.get("year", 2024),
                    value=Decimal(str(dp.get("value", 0))),
                    predicted=dp.get("predicted", False),
                    confidence_lower=Decimal(str(dp.get("ci_lower", 0))) if dp.get("ci_lower") else None,
                    confidence_upper=Decimal(str(dp.get("ci_upper", 0))) if dp.get("ci_upper") else None,
                )
            )

        slope = Decimal(str(result.get("slope", 0)))
        r_squared = Decimal(str(result.get("r_squared", 0)))
        total_change = Decimal(str(result.get("total_change", 0)))
        annualized = Decimal(str(result.get("annualized_change", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"trend:{body.country_code}:{body.index_type}", str(slope)
        )

        logger.info(
            "Trend analysis completed: country=%s index=%s slope=%s r2=%s operator=%s",
            body.country_code,
            body.index_type,
            slope,
            r_squared,
            user.operator_id or user.user_id,
        )

        return TrendAnalysisResponse(
            country_code=body.country_code,
            country_name=result.get("country_name", ""),
            index_type=body.index_type,
            trend_direction=TrendDirectionEnum(result.get("trend_direction", "stable")),
            slope=slope,
            r_squared=r_squared,
            trend_reliable=result.get("trend_reliable", False),
            data_points=data_points,
            period_start=result.get("period_start", body.start_year or 2015),
            period_end=result.get("period_end", body.end_year or 2024),
            total_change=total_change,
            annualized_change=annualized,
            trend_reversal_detected=result.get("trend_reversal_detected", False),
            reversal_year=result.get("reversal_year"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Trend analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trend analysis failed",
        )


# ---------------------------------------------------------------------------
# GET /trends/{country_code}/trajectory
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/trajectory",
    response_model=TrajectoryResponse,
    summary="Get corruption trajectory for a country",
    description=(
        "Retrieve the current corruption trajectory for a country including "
        "direction, strength, momentum, acceleration, and phase classification "
        "(accelerating, decelerating, stable, inflecting)."
    ),
    responses={
        200: {"description": "Trajectory retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_trajectory(
    country_code: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:trends:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TrajectoryResponse:
    """Get the corruption trajectory for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        user: Authenticated user with trends:read permission.

    Returns:
        TrajectoryResponse with trajectory metrics.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_trend_engine()
        result = engine.get_trajectory(normalized_code)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trajectory data not found for {normalized_code}",
            )

        momentum = Decimal(str(result.get("momentum", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"trajectory:{normalized_code}", str(momentum)
        )

        logger.info(
            "Trajectory retrieved: country=%s direction=%s momentum=%s operator=%s",
            normalized_code,
            result.get("trajectory_direction", "stable"),
            momentum,
            user.operator_id or user.user_id,
        )

        cpi_traj = result.get("cpi_trajectory")
        wgi_traj = result.get("wgi_trajectory")

        return TrajectoryResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            trajectory_direction=TrendDirectionEnum(result.get("trajectory_direction", "stable")),
            trajectory_strength=Decimal(str(result.get("trajectory_strength", 0))),
            cpi_trajectory=TrendDataPoint(
                year=cpi_traj.get("year", 2024),
                value=Decimal(str(cpi_traj.get("value", 0))),
            ) if cpi_traj else None,
            wgi_trajectory=TrendDataPoint(
                year=wgi_traj.get("year", 2024),
                value=Decimal(str(wgi_traj.get("value", 0))),
            ) if wgi_traj else None,
            momentum=momentum,
            acceleration=Decimal(str(result.get("acceleration", 0))),
            phase=result.get("phase", "stable"),
            risk_outlook=RiskLevelEnum(result.get("risk_outlook", "moderate")),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Trajectory retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trajectory retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /trends/prediction
# ---------------------------------------------------------------------------


@router.post(
    "/prediction",
    response_model=PredictionResponse,
    summary="Predict future corruption index values",
    description=(
        "Generate predictions for future corruption index values based on "
        "historical trend analysis. Includes confidence intervals and "
        "reliability assessment. Default prediction horizon is 3 years."
    ),
    responses={
        200: {"description": "Prediction completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def predict_trends(
    request: Request,
    body: PredictionRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:trends:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PredictionResponse:
    """Predict future corruption index values for a country.

    Args:
        body: Prediction request with country, index type, and horizon.
        user: Authenticated user with trends:create permission.

    Returns:
        PredictionResponse with predicted values and confidence intervals.
    """
    start = time.monotonic()

    try:
        engine = get_trend_engine()
        result = engine.predict(
            country_code=body.country_code,
            index_type=body.index_type,
            horizon_years=body.horizon_years,
            confidence_level=float(body.confidence_level),
        )

        predictions = []
        for dp in result.get("predictions", []):
            predictions.append(
                TrendDataPoint(
                    year=dp.get("year", 2025),
                    value=Decimal(str(dp.get("value", 0))),
                    predicted=True,
                    confidence_lower=Decimal(str(dp.get("ci_lower", 0))) if dp.get("ci_lower") else None,
                    confidence_upper=Decimal(str(dp.get("ci_upper", 0))) if dp.get("ci_upper") else None,
                )
            )

        r_squared = Decimal(str(result.get("model_r_squared", 0)))

        if r_squared >= Decimal("0.7"):
            confidence = PredictionConfidenceEnum.HIGH
        elif r_squared >= Decimal("0.4"):
            confidence = PredictionConfidenceEnum.MEDIUM
        elif r_squared >= Decimal("0.2"):
            confidence = PredictionConfidenceEnum.LOW
        else:
            confidence = PredictionConfidenceEnum.UNRELIABLE

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"prediction:{body.country_code}:{body.index_type}:{body.horizon_years}",
            str(r_squared),
        )

        logger.info(
            "Prediction completed: country=%s index=%s horizon=%d confidence=%s operator=%s",
            body.country_code,
            body.index_type,
            body.horizon_years,
            confidence.value,
            user.operator_id or user.user_id,
        )

        return PredictionResponse(
            country_code=body.country_code,
            country_name=result.get("country_name", ""),
            index_type=body.index_type,
            predictions=predictions,
            prediction_confidence=confidence,
            model_r_squared=r_squared,
            base_year=result.get("base_year", 2024),
            base_value=Decimal(str(result.get("base_value", 0))),
            predicted_risk_trajectory=TrendDirectionEnum(result.get("predicted_trajectory", "stable")),
            warning_flags=result.get("warning_flags", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["Transparency International CPI", "World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trend prediction failed",
        )


# ---------------------------------------------------------------------------
# GET /trends/improving
# ---------------------------------------------------------------------------


@router.get(
    "/improving",
    response_model=ImprovingCountriesResponse,
    summary="Get countries with improving corruption trends",
    description=(
        "Retrieve countries showing significant improvement in corruption "
        "indices over a specified period, sorted by improvement magnitude."
    ),
    responses={
        200: {"description": "Improving countries retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_improving_countries(
    request: Request,
    index_type: str = Query(default="cpi", description="Index type: cpi, wgi, or composite"),
    period_years: int = Query(default=5, ge=2, le=20, description="Analysis period in years"),
    region: Optional[RegionEnum] = Query(None, description="Region filter"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:trends:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ImprovingCountriesResponse:
    """Get countries with improving corruption trends.

    Args:
        index_type: Index type to analyze.
        period_years: Analysis period in years.
        region: Optional region filter.
        pagination: Pagination parameters.
        user: Authenticated user with trends:read permission.

    Returns:
        ImprovingCountriesResponse with improving countries list.
    """
    start = time.monotonic()

    try:
        engine = get_trend_engine()
        result = engine.get_improving_countries(
            index_type=index_type,
            period_years=period_years,
            region=region.value if region else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        countries = []
        for entry in result.get("countries", []):
            countries.append(
                TrendCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    current_score=Decimal(str(entry.get("current_score", 0))),
                    previous_score=Decimal(str(entry.get("previous_score", 0))),
                    change=Decimal(str(entry.get("change", 0))),
                    change_percent=Decimal(str(entry.get("change_percent", 0))),
                    trend_direction=TrendDirectionEnum.IMPROVING,
                    region=RegionEnum(entry.get("region")) if entry.get("region") else None,
                )
            )

        total = result.get("total_improving", len(countries))
        period_desc = result.get("period", f"{2024 - period_years}-2024")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"improving:{index_type}:{period_years}", str(total)
        )

        logger.info(
            "Improving countries retrieved: index=%s period=%dy total=%d operator=%s",
            index_type,
            period_years,
            total,
            user.operator_id or user.user_id,
        )

        return ImprovingCountriesResponse(
            period=period_desc,
            index_type=index_type,
            countries=countries,
            total_improving=total,
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
                data_sources=["Transparency International CPI", "World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Improving countries retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Improving countries retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /trends/deteriorating
# ---------------------------------------------------------------------------


@router.get(
    "/deteriorating",
    response_model=DeterioratingCountriesResponse,
    summary="Get countries with deteriorating corruption trends",
    description=(
        "Retrieve countries showing significant deterioration in corruption "
        "indices, sorted by deterioration magnitude. Includes overlap count "
        "with EUDR high-risk classification."
    ),
    responses={
        200: {"description": "Deteriorating countries retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_deteriorating_countries(
    request: Request,
    index_type: str = Query(default="cpi", description="Index type: cpi, wgi, or composite"),
    period_years: int = Query(default=5, ge=2, le=20, description="Analysis period in years"),
    region: Optional[RegionEnum] = Query(None, description="Region filter"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:trends:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DeterioratingCountriesResponse:
    """Get countries with deteriorating corruption trends.

    Args:
        index_type: Index type to analyze.
        period_years: Analysis period in years.
        region: Optional region filter.
        pagination: Pagination parameters.
        user: Authenticated user with trends:read permission.

    Returns:
        DeterioratingCountriesResponse with deteriorating countries list.
    """
    start = time.monotonic()

    try:
        engine = get_trend_engine()
        result = engine.get_deteriorating_countries(
            index_type=index_type,
            period_years=period_years,
            region=region.value if region else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        countries = []
        for entry in result.get("countries", []):
            countries.append(
                TrendCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    current_score=Decimal(str(entry.get("current_score", 0))),
                    previous_score=Decimal(str(entry.get("previous_score", 0))),
                    change=Decimal(str(entry.get("change", 0))),
                    change_percent=Decimal(str(entry.get("change_percent", 0))),
                    trend_direction=TrendDirectionEnum.DETERIORATING,
                    region=RegionEnum(entry.get("region")) if entry.get("region") else None,
                )
            )

        total = result.get("total_deteriorating", len(countries))
        eudr_overlap = result.get("eudr_high_risk_overlap", 0)
        period_desc = result.get("period", f"{2024 - period_years}-2024")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"deteriorating:{index_type}:{period_years}", str(total)
        )

        logger.info(
            "Deteriorating countries retrieved: index=%s total=%d eudr_overlap=%d operator=%s",
            index_type,
            total,
            eudr_overlap,
            user.operator_id or user.user_id,
        )

        return DeterioratingCountriesResponse(
            period=period_desc,
            index_type=index_type,
            countries=countries,
            total_deteriorating=total,
            eudr_high_risk_overlap=eudr_overlap,
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
                data_sources=["Transparency International CPI", "World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Deteriorating countries retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deteriorating countries retrieval failed",
        )
