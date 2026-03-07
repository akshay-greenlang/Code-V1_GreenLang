# -*- coding: utf-8 -*-
"""
Analysis Routes - AGENT-EUDR-003 Satellite Monitoring API

Endpoints for satellite analysis operations including spectral index
calculation, baseline establishment, change detection, and multi-source
fusion analysis.

Endpoints:
    POST /ndvi               - Calculate spectral index
    POST /baseline           - Establish Dec 2020 baseline
    GET  /baseline/{plot_id} - Get stored baseline
    POST /change-detect      - Run change detection
    POST /fusion             - Run multi-source data fusion
    GET  /history/{plot_id}  - Get analysis history

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.satellite_monitoring.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_baseline_manager,
    get_change_detector,
    get_fusion_engine,
    get_pagination,
    get_spectral_calculator,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    BaselineApiResponse,
    CalculateIndexApiRequest,
    ChangeDetectionApiResponse,
    DetectChangeApiRequest,
    EstablishBaselineApiRequest,
    FusionApiRequest,
    FusionApiResponse,
    PaginatedMeta,
    SpectralIndexApiResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Satellite Analysis"])


# ---------------------------------------------------------------------------
# POST /ndvi
# ---------------------------------------------------------------------------


@router.post(
    "/ndvi",
    response_model=SpectralIndexApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate spectral vegetation index",
    description=(
        "Calculate a spectral vegetation index (NDVI, EVI, NBR, NDMI, "
        "SAVI, MSAVI) from provided band reflectance data. Returns the "
        "index statistics, vegetation classification, and optional full "
        "index array for small areas."
    ),
    responses={
        200: {"description": "Spectral index result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def calculate_spectral_index(
    body: CalculateIndexApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SpectralIndexApiResponse:
    """Calculate a spectral vegetation index from band data.

    Computes the requested index from red and NIR bands (with optional
    SWIR and blue bands for NBR/NDMI/EVI). Returns statistics and a
    biome-aware classification.

    Args:
        body: Calculation request with band arrays and index type.
        user: Authenticated user with analysis:write permission.

    Returns:
        SpectralIndexApiResponse with index values and classification.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Spectral index calculation: user=%s index_type=%s biome=%s "
        "pixels=%dx%d",
        user.user_id,
        body.index_type,
        body.biome,
        len(body.red_band),
        len(body.red_band[0]) if body.red_band else 0,
    )

    try:
        calculator = get_spectral_calculator()

        result = calculator.calculate(
            red_band=body.red_band,
            nir_band=body.nir_band,
            index_type=body.index_type,
            biome=body.biome,
            swir_band=body.swir_band,
            blue_band=body.blue_band,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Spectral index calculated: user=%s index=%s mean=%.4f "
            "classification=%s elapsed_ms=%.1f",
            user.user_id,
            body.index_type,
            getattr(result, "mean_value", 0.0),
            getattr(result, "classification", "unknown"),
            elapsed * 1000,
        )

        return SpectralIndexApiResponse(
            index_type=body.index_type,
            mean_value=getattr(result, "mean_value", 0.0),
            min_value=getattr(result, "min_value", 0.0),
            max_value=getattr(result, "max_value", 0.0),
            std_dev=getattr(result, "std_dev", 0.0),
            pixel_count=getattr(result, "pixel_count", 0),
            classification=getattr(result, "classification", "unknown"),
            biome_threshold_used=getattr(result, "biome_threshold_used", None),
            index_values=getattr(result, "index_values", None),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Spectral index error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Spectral index calculation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spectral index calculation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /baseline
# ---------------------------------------------------------------------------


@router.post(
    "/baseline",
    response_model=BaselineApiResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Establish Dec 2020 baseline for a plot",
    description=(
        "Establish a spectral baseline snapshot for a production plot "
        "using satellite imagery from around the EUDR cutoff date "
        "(December 31, 2020). Composites cloud-free scenes to determine "
        "baseline NDVI, forest cover percentage, and canopy density."
    ),
    responses={
        201: {"description": "Baseline established"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def establish_baseline(
    body: EstablishBaselineApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BaselineApiResponse:
    """Establish a Dec 2020 baseline snapshot for a plot.

    Creates a baseline using composited satellite imagery around
    the EUDR cutoff date. The baseline includes NDVI, forest cover,
    and canopy density metrics for comparison in change detection.

    Args:
        body: Baseline request with plot details and polygon.
        user: Authenticated user with analysis:write permission.

    Returns:
        BaselineApiResponse with baseline metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    baseline_id = f"bsl-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Baseline establishment: user=%s plot_id=%s commodity=%s "
        "country=%s vertices=%d",
        user.user_id,
        body.plot_id,
        body.commodity,
        body.country_code,
        len(body.polygon_vertices),
    )

    try:
        manager = get_baseline_manager()

        result = manager.establish_baseline(
            plot_id=body.plot_id,
            polygon_vertices=body.polygon_vertices,
            commodity=body.commodity,
            country_code=body.country_code,
            biome=body.biome,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Baseline established: user=%s plot_id=%s baseline_ndvi=%.4f "
            "forest_cover=%.1f%% scenes=%d elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "baseline_ndvi", 0.0),
            getattr(result, "forest_cover_pct", 0.0),
            getattr(result, "scenes_used", 0),
            elapsed * 1000,
        )

        return BaselineApiResponse(
            baseline_id=getattr(result, "baseline_id", baseline_id),
            plot_id=body.plot_id,
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            baseline_ndvi=getattr(result, "baseline_ndvi", 0.0),
            baseline_evi=getattr(result, "baseline_evi", None),
            forest_cover_pct=getattr(result, "forest_cover_pct", 0.0),
            canopy_density_pct=getattr(result, "canopy_density_pct", None),
            area_ha=getattr(result, "area_ha", 0.0),
            commodity=body.commodity,
            country_code=body.country_code,
            biome=body.biome,
            scenes_used=getattr(result, "scenes_used", 0),
            cloud_free_coverage_pct=getattr(result, "cloud_free_coverage_pct", 0.0),
            data_sources=getattr(result, "data_sources", []),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Baseline establishment error: user=%s plot_id=%s error=%s",
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
            "Baseline establishment failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline establishment failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /baseline/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/baseline/{plot_id}",
    response_model=BaselineApiResponse,
    summary="Get stored baseline for a plot",
    description="Retrieve the stored baseline snapshot for a production plot.",
    responses={
        200: {"description": "Baseline snapshot"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Baseline not found"},
    },
)
async def get_baseline(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BaselineApiResponse:
    """Retrieve the stored baseline snapshot for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with analysis:read permission.

    Returns:
        BaselineApiResponse with stored baseline data.

    Raises:
        HTTPException: 404 if baseline not found.
    """
    logger.info(
        "Baseline retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        manager = get_baseline_manager()
        result = manager.get_baseline(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Baseline for plot {plot_id} not found",
            )

        return BaselineApiResponse(
            baseline_id=getattr(result, "baseline_id", ""),
            plot_id=plot_id,
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            baseline_ndvi=getattr(result, "baseline_ndvi", 0.0),
            baseline_evi=getattr(result, "baseline_evi", None),
            forest_cover_pct=getattr(result, "forest_cover_pct", 0.0),
            canopy_density_pct=getattr(result, "canopy_density_pct", None),
            area_ha=getattr(result, "area_ha", 0.0),
            commodity=getattr(result, "commodity", ""),
            country_code=getattr(result, "country_code", ""),
            biome=getattr(result, "biome", None),
            scenes_used=getattr(result, "scenes_used", 0),
            cloud_free_coverage_pct=getattr(result, "cloud_free_coverage_pct", 0.0),
            data_sources=getattr(result, "data_sources", []),
            provenance_hash=getattr(result, "provenance_hash", ""),
            established_at=getattr(result, "established_at", datetime.now(timezone.utc)),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Baseline retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /change-detect
# ---------------------------------------------------------------------------


@router.post(
    "/change-detect",
    response_model=ChangeDetectionApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Run deforestation change detection",
    description=(
        "Run multi-method deforestation change detection on a production "
        "plot by comparing current satellite imagery against the EUDR "
        "cutoff baseline. Supports quick, standard, and deep analysis "
        "levels with increasing accuracy and processing time."
    ),
    responses={
        200: {"description": "Change detection result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_change(
    body: DetectChangeApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ChangeDetectionApiResponse:
    """Run change detection on a production plot.

    Compares current satellite imagery against the EUDR cutoff baseline
    to detect deforestation, degradation, or regrowth. Returns
    classification, confidence, and forest loss estimates.

    Args:
        body: Detection request with plot details and analysis parameters.
        user: Authenticated user with analysis:write permission.

    Returns:
        ChangeDetectionApiResponse with detection results.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    detection_id = f"det-{uuid.uuid4().hex[:12]}"
    analysis_target = body.analysis_date or date.today()

    logger.info(
        "Change detection: user=%s plot_id=%s commodity=%s "
        "analysis_date=%s level=%s",
        user.user_id,
        body.plot_id,
        body.commodity,
        analysis_target,
        body.analysis_level,
    )

    try:
        detector = get_change_detector()

        result = detector.detect_change(
            plot_id=body.plot_id,
            polygon_vertices=body.polygon_vertices,
            commodity=body.commodity,
            analysis_date=analysis_target,
            analysis_level=body.analysis_level,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Change detection completed: user=%s plot_id=%s "
            "deforestation=%s classification=%s confidence=%.2f "
            "loss_ha=%.3f elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "deforestation_detected", False),
            getattr(result, "change_classification", "no_change"),
            getattr(result, "confidence", 0.0),
            getattr(result, "forest_loss_ha", 0.0),
            elapsed * 1000,
        )

        return ChangeDetectionApiResponse(
            detection_id=getattr(result, "detection_id", detection_id),
            plot_id=body.plot_id,
            deforestation_detected=getattr(result, "deforestation_detected", False),
            change_classification=getattr(result, "change_classification", "no_change"),
            ndvi_baseline=getattr(result, "ndvi_baseline", 0.0),
            ndvi_current=getattr(result, "ndvi_current", 0.0),
            ndvi_delta=getattr(result, "ndvi_delta", 0.0),
            forest_loss_ha=getattr(result, "forest_loss_ha", 0.0),
            forest_loss_pct=getattr(result, "forest_loss_pct", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            analysis_level=body.analysis_level,
            analysis_date=analysis_target,
            cutoff_date=getattr(result, "cutoff_date", "2020-12-31"),
            data_sources=getattr(result, "data_sources", []),
            change_pixels=getattr(result, "change_pixels", None),
            total_pixels=getattr(result, "total_pixels", None),
            seasonal_adjusted=getattr(result, "seasonal_adjusted", False),
            alerts_generated=getattr(result, "alerts_generated", 0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Change detection error: user=%s plot_id=%s error=%s",
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
            "Change detection failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Change detection failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /fusion
# ---------------------------------------------------------------------------


@router.post(
    "/fusion",
    response_model=FusionApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Run multi-source data fusion",
    description=(
        "Fuse analysis results from multiple satellite sources (Sentinel-2, "
        "Landsat, GFW) using weighted combination. Returns a single fused "
        "detection result with cross-source agreement metrics."
    ),
    responses={
        200: {"description": "Fusion result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def run_fusion(
    body: FusionApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FusionApiResponse:
    """Run multi-source data fusion analysis.

    Combines results from Sentinel-2, Landsat, and GFW sources
    using weighted fusion to produce a single reliable detection
    result with cross-source agreement metrics.

    Args:
        body: Fusion request with per-source results.
        user: Authenticated user with analysis:write permission.

    Returns:
        FusionApiResponse with fused detection result.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    fusion_id = f"fus-{uuid.uuid4().hex[:12]}"

    # Count provided sources
    sources_provided = sum([
        body.sentinel2_result is not None,
        body.landsat_result is not None,
        body.gfw_result is not None,
    ])

    if sources_provided == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one source result (sentinel2, landsat, or gfw) is required",
        )

    logger.info(
        "Fusion analysis: user=%s plot_id=%s sources=%d",
        user.user_id,
        body.plot_id,
        sources_provided,
    )

    try:
        engine = get_fusion_engine()

        source_results = {}
        if body.sentinel2_result is not None:
            source_results["sentinel2"] = body.sentinel2_result.model_dump()
        if body.landsat_result is not None:
            source_results["landsat"] = body.landsat_result.model_dump()
        if body.gfw_result is not None:
            source_results["gfw"] = body.gfw_result.model_dump()

        result = engine.fuse(
            plot_id=body.plot_id,
            source_results=source_results,
            custom_weights=body.custom_weights,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Fusion completed: user=%s plot_id=%s fused_detected=%s "
            "confidence=%.2f agreement=%.2f elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "fused_deforestation_detected", False),
            getattr(result, "fused_confidence", 0.0),
            getattr(result, "source_agreement", 0.0),
            elapsed * 1000,
        )

        return FusionApiResponse(
            fusion_id=getattr(result, "fusion_id", fusion_id),
            plot_id=body.plot_id,
            fused_deforestation_detected=getattr(
                result, "fused_deforestation_detected", False
            ),
            fused_confidence=getattr(result, "fused_confidence", 0.0),
            fused_ndvi_delta=getattr(result, "fused_ndvi_delta", None),
            fused_forest_loss_ha=getattr(result, "fused_forest_loss_ha", 0.0),
            sources_used=sources_provided,
            source_agreement=getattr(result, "source_agreement", 0.0),
            weights_applied=getattr(result, "weights_applied", {}),
            per_source_summary=getattr(result, "per_source_summary", []),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Fusion error: user=%s plot_id=%s error=%s",
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
            "Fusion failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fusion analysis failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /history/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/history/{plot_id}",
    response_model=Dict[str, Any],
    summary="Get analysis history for a plot",
    description=(
        "Retrieve paginated change detection history for a production "
        "plot. Returns historical detection results sorted by analysis "
        "date descending."
    ),
    responses={
        200: {"description": "Analysis history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def get_analysis_history(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:analysis:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get paginated analysis history for a plot.

    Args:
        plot_id: Plot identifier.
        pagination: Pagination parameters.
        user: Authenticated user with analysis:read permission.

    Returns:
        Dictionary with analysis history items and pagination metadata.

    Raises:
        HTTPException: 404 if plot not found, 500 on error.
    """
    logger.info(
        "Analysis history: user=%s plot_id=%s limit=%d offset=%d",
        user.user_id,
        plot_id,
        pagination.limit,
        pagination.offset,
    )

    try:
        detector = get_change_detector()

        history = detector.get_history(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis history found for plot {plot_id}",
            )

        items = getattr(history, "items", [])
        total = getattr(history, "total", len(items))

        return {
            "plot_id": plot_id,
            "items": [
                {
                    "detection_id": getattr(item, "detection_id", ""),
                    "analysis_date": str(getattr(item, "analysis_date", "")),
                    "deforestation_detected": getattr(item, "deforestation_detected", False),
                    "change_classification": getattr(item, "change_classification", "no_change"),
                    "ndvi_delta": getattr(item, "ndvi_delta", 0.0),
                    "confidence": getattr(item, "confidence", 0.0),
                    "forest_loss_ha": getattr(item, "forest_loss_ha", 0.0),
                }
                for item in items
            ],
            "meta": {
                "total": total,
                "limit": pagination.limit,
                "offset": pagination.offset,
                "has_more": (pagination.offset + pagination.limit) < total,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Analysis history failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis history retrieval failed due to an internal error",
        )
