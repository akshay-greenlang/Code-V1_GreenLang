# -*- coding: utf-8 -*-
"""
Analysis Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for canopy height estimation, forest fragmentation analysis,
above-ground biomass estimation, complete plot profile retrieval, and
multi-metric temporal comparison.

Endpoints:
    POST /height           - Estimate canopy height
    POST /fragmentation    - Analyze forest fragmentation
    POST /biomass          - Estimate above-ground biomass
    GET  /{plot_id}/profile - Get complete plot profile
    POST /compare          - Compare metrics between two dates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_biomass_engine,
    get_forest_cover_service,
    get_fragmentation_engine,
    get_height_engine,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    AnalysisComparisonResponse,
    AnalyzeFragmentationRequest,
    BiomassResponse,
    CanopyHeightResponse,
    CompareAnalysisRequest,
    EstimateBiomassRequest,
    EstimateHeightRequest,
    FragmentationResponse,
    MetricComparison,
    PlotProfileResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forest Analysis"])


# ---------------------------------------------------------------------------
# POST /height
# ---------------------------------------------------------------------------


@router.post(
    "/height",
    response_model=CanopyHeightResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate canopy height",
    description=(
        "Estimate canopy height for a production plot using GEDI, "
        "ICESat-2, LiDAR, photogrammetry, or radar data sources. "
        "Returns height statistics including mean, median, max, P95, "
        "and FAO threshold compliance (>=5m)."
    ),
    responses={
        200: {"description": "Canopy height estimation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def estimate_height(
    body: EstimateHeightRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CanopyHeightResponse:
    """Estimate canopy height for a production plot.

    Uses the specified height data sources to estimate canopy height
    statistics across the plot area.

    Args:
        body: Height estimation request with plot polygon and sources.
        user: Authenticated user with analysis:write permission.

    Returns:
        CanopyHeightResponse with height statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    sources = (
        [s.value for s in body.sources]
        if body.sources
        else ["gedi", "icesat2"]
    )

    logger.info(
        "Height estimation: user=%s plot_id=%s sources=%s",
        user.user_id,
        body.plot_id,
        sources,
    )

    try:
        engine = get_height_engine()

        result = engine.estimate(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            sources=sources,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Height estimation completed: user=%s plot_id=%s "
            "mean=%.1fm max=%.1fm fao_met=%s elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "mean_height_m", 0.0),
            getattr(result, "max_height_m", 0.0),
            getattr(result, "fao_threshold_met", False),
            elapsed * 1000,
        )

        return CanopyHeightResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            mean_height_m=getattr(result, "mean_height_m", 0.0),
            median_height_m=getattr(result, "median_height_m", 0.0),
            max_height_m=getattr(result, "max_height_m", 0.0),
            min_height_m=getattr(result, "min_height_m", 0.0),
            std_dev_m=getattr(result, "std_dev_m", 0.0),
            p95_height_m=getattr(result, "p95_height_m", 0.0),
            fao_threshold_met=getattr(result, "fao_threshold_met", False),
            height_distribution=getattr(result, "height_distribution", None),
            sources_used=sources,
            footprint_count=getattr(result, "footprint_count", 0),
            area_ha=getattr(result, "area_ha", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Height estimation error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Height estimation failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Height estimation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /fragmentation
# ---------------------------------------------------------------------------


@router.post(
    "/fragmentation",
    response_model=FragmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze forest fragmentation",
    description=(
        "Analyze forest fragmentation metrics for a production plot. "
        "Computes patch count, size distribution, edge density, core "
        "area percentage, shape index, connectivity, and fragmentation "
        "classification (intact, perforated, fragmented, patch, relictual)."
    ),
    responses={
        200: {"description": "Fragmentation analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_fragmentation(
    body: AnalyzeFragmentationRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FragmentationResponse:
    """Analyze forest fragmentation for a production plot.

    Computes landscape-level fragmentation metrics using the specified
    edge buffer distance.

    Args:
        body: Fragmentation request with plot polygon and edge buffer.
        user: Authenticated user with analysis:write permission.

    Returns:
        FragmentationResponse with fragmentation metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Fragmentation analysis: user=%s plot_id=%s edge_buffer=%.0fm",
        user.user_id,
        body.plot_id,
        body.edge_buffer_m,
    )

    try:
        engine = get_fragmentation_engine()

        result = engine.analyze(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            edge_buffer_m=body.edge_buffer_m,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Fragmentation analysis completed: user=%s plot_id=%s "
            "patches=%d class=%s core=%.1f%% elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "total_patches", 0),
            getattr(result, "fragmentation_class", "unknown"),
            getattr(result, "core_area_pct", 0.0),
            elapsed * 1000,
        )

        return FragmentationResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            total_patches=getattr(result, "total_patches", 0),
            largest_patch_ha=getattr(result, "largest_patch_ha", 0.0),
            mean_patch_ha=getattr(result, "mean_patch_ha", 0.0),
            edge_density_m_per_ha=getattr(
                result, "edge_density_m_per_ha", 0.0
            ),
            core_area_pct=getattr(result, "core_area_pct", 0.0),
            edge_area_pct=getattr(result, "edge_area_pct", 0.0),
            shape_index=getattr(result, "shape_index", 0.0),
            connectivity_index=getattr(result, "connectivity_index", 0.0),
            fragmentation_class=getattr(
                result, "fragmentation_class", "unknown"
            ),
            edge_buffer_m=body.edge_buffer_m,
            area_ha=getattr(result, "area_ha", 0.0),
            forest_area_ha=getattr(result, "forest_area_ha", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Fragmentation error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Fragmentation failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fragmentation analysis failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /biomass
# ---------------------------------------------------------------------------


@router.post(
    "/biomass",
    response_model=BiomassResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate above-ground biomass",
    description=(
        "Estimate above-ground biomass (AGB) and carbon stock for a "
        "production plot using GEDI L4A, ESA CCI Biomass, GlobBiomass, "
        "LiDAR, or allometric models. Returns AGB in tonnes per hectare "
        "and estimated carbon stock."
    ),
    responses={
        200: {"description": "Biomass estimation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def estimate_biomass(
    body: EstimateBiomassRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BiomassResponse:
    """Estimate above-ground biomass for a production plot.

    Uses the specified biomass data sources to estimate AGB and
    carbon stock across the plot area.

    Args:
        body: Biomass estimation request with plot polygon and sources.
        user: Authenticated user with analysis:write permission.

    Returns:
        BiomassResponse with biomass and carbon metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    sources = (
        [s.value for s in body.sources]
        if body.sources
        else ["gedi_l4a", "esa_cci"]
    )

    logger.info(
        "Biomass estimation: user=%s plot_id=%s sources=%s",
        user.user_id,
        body.plot_id,
        sources,
    )

    try:
        engine = get_biomass_engine()

        result = engine.estimate(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            sources=sources,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Biomass estimation completed: user=%s plot_id=%s "
            "agb_mean=%.1f t/ha carbon=%.1f t/ha elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "agb_mean_t_ha", 0.0),
            getattr(result, "carbon_stock_t_ha", 0.0),
            elapsed * 1000,
        )

        return BiomassResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            agb_mean_t_ha=getattr(result, "agb_mean_t_ha", 0.0),
            agb_total_t=getattr(result, "agb_total_t", 0.0),
            agb_median_t_ha=getattr(result, "agb_median_t_ha", 0.0),
            agb_std_dev_t_ha=getattr(result, "agb_std_dev_t_ha", 0.0),
            carbon_stock_t_ha=getattr(result, "carbon_stock_t_ha", 0.0),
            carbon_stock_total_t=getattr(result, "carbon_stock_total_t", 0.0),
            sources_used=sources,
            area_ha=getattr(result, "area_ha", 0.0),
            sample_count=getattr(result, "sample_count", 0),
            uncertainty_pct=getattr(result, "uncertainty_pct", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Biomass estimation error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Biomass estimation failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Biomass estimation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}/profile
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/profile",
    response_model=PlotProfileResponse,
    summary="Get complete plot profile",
    description=(
        "Retrieve the complete stored analysis profile for a production "
        "plot including all available results: density, classification, "
        "historical, height, fragmentation, biomass, and verification."
    ),
    responses={
        200: {"description": "Complete plot profile"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Profile not found"},
    },
)
async def get_plot_profile(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:analysis:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PlotProfileResponse:
    """Get the complete analysis profile for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with analysis:read permission.

    Returns:
        PlotProfileResponse with all available analysis results.

    Raises:
        HTTPException: 404 if profile not found.
    """
    start = time.monotonic()

    logger.info(
        "Plot profile retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        service = get_forest_cover_service()
        result = service.get_profile(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis profile found for plot {plot_id}",
            )

        elapsed = time.monotonic() - start

        return PlotProfileResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            area_ha=getattr(result, "area_ha", 0.0),
            density=getattr(result, "density", None),
            classification=getattr(result, "classification", None),
            historical=getattr(result, "historical", None),
            height=getattr(result, "height", None),
            fragmentation=getattr(result, "fragmentation", None),
            biomass=getattr(result, "biomass", None),
            verification=getattr(result, "verification", None),
            fao_forest_status=getattr(
                result, "fao_forest_status", "unknown"
            ),
            overall_confidence=getattr(result, "overall_confidence", 0.0),
            last_updated=getattr(
                result, "last_updated", datetime.now(timezone.utc)
            ),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Plot profile failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plot profile retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=AnalysisComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare metrics between two dates",
    description=(
        "Compare canopy height, biomass, and fragmentation metrics "
        "between two dates for the same plot. Returns per-metric "
        "absolute and relative changes with an overall direction."
    ),
    responses={
        200: {"description": "Metric comparison result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_metrics(
    body: CompareAnalysisRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AnalysisComparisonResponse:
    """Compare analysis metrics between two dates for a plot.

    Args:
        body: Comparison request with plot polygon, dates, and metrics.
        user: Authenticated user with analysis:write permission.

    Returns:
        AnalysisComparisonResponse with per-metric comparisons.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    metrics = body.metrics or ["height", "biomass", "fragmentation"]

    logger.info(
        "Metric comparison: user=%s plot_id=%s before=%s after=%s "
        "metrics=%s",
        user.user_id,
        body.plot_id,
        body.date_before,
        body.date_after,
        metrics,
    )

    try:
        service = get_forest_cover_service()

        result = service.compare_metrics(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            date_before=body.date_before,
            date_after=body.date_after,
            metrics=metrics,
        )

        # Build comparison items
        comparisons = []
        raw_comparisons = getattr(result, "comparisons", [])
        for comp in raw_comparisons:
            comparisons.append(MetricComparison(
                metric_name=getattr(comp, "metric_name", ""),
                value_before=getattr(comp, "value_before", 0.0),
                value_after=getattr(comp, "value_after", 0.0),
                absolute_change=getattr(comp, "absolute_change", 0.0),
                relative_change_pct=getattr(
                    comp, "relative_change_pct", 0.0
                ),
                unit=getattr(comp, "unit", ""),
            ))

        elapsed = time.monotonic() - start
        logger.info(
            "Metric comparison completed: user=%s plot_id=%s "
            "direction=%s metrics=%d elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "overall_change_direction", "stable"),
            len(comparisons),
            elapsed * 1000,
        )

        return AnalysisComparisonResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            date_before=body.date_before,
            date_after=body.date_after,
            comparisons=comparisons,
            overall_change_direction=getattr(
                result, "overall_change_direction", "stable"
            ),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Metric comparison error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Metric comparison failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metric comparison failed due to an internal error",
        )
