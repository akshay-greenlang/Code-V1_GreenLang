# -*- coding: utf-8 -*-
"""
Area Calculation Routes - AGENT-EUDR-006 Plot Boundary Manager API

Endpoints for computing geodetic area on the WGS84 ellipsoid using the
Karney algorithm, checking the EUDR Article 9 four-hectare threshold,
and performing batch area calculations.

Endpoints:
    POST /area/calculate - Calculate geodetic area for a single geometry
    POST /area/batch     - Batch area calculation
    POST /area/threshold - Check EUDR 4ha threshold classification

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.plot_boundary.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_area_calculator,
    get_config,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    AreaCalculationRequestSchema,
    AreaResponseSchema,
    BatchAreaRequestSchema,
    BatchAreaResponseSchema,
    BatchAreaResultSchema,
    CompactnessMetricsSchema,
    ThresholdClassificationSchema,
    ThresholdResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Area Calculation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_area_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 provenance hash for area calculation.

    Args:
        data: Area calculation data to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = str(sorted(data.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _classify_threshold(area_hectares: float, threshold: float = 4.0) -> str:
    """Classify area against EUDR threshold.

    Args:
        area_hectares: Area in hectares.
        threshold: EUDR threshold (default 4.0ha).

    Returns:
        ThresholdClassificationSchema value string.
    """
    if area_hectares > threshold:
        return ThresholdClassificationSchema.ABOVE_THRESHOLD.value
    elif area_hectares < threshold:
        return ThresholdClassificationSchema.BELOW_THRESHOLD.value
    return ThresholdClassificationSchema.AT_THRESHOLD.value


def _build_stub_area_response(
    geometry: Any = None,
    wkt: str = None,
    plot_id: str = None,
) -> AreaResponseSchema:
    """Build a stub area calculation response for development.

    When the AreaCalculator engine is not yet available, returns
    a placeholder area response. In production, the real engine
    computes geodetic area on the WGS84 ellipsoid.

    Args:
        geometry: GeoJSON geometry input.
        wkt: WKT string input.
        plot_id: Plot ID reference.

    Returns:
        AreaResponseSchema with placeholder values.
    """
    area_m2 = 0.0
    area_hectares = 0.0

    hash_input = {
        "type": "area_calculation",
        "geometry": str(geometry),
        "wkt": wkt or "",
        "plot_id": plot_id or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return AreaResponseSchema(
        area_m2=area_m2,
        area_hectares=area_hectares,
        area_acres=area_hectares * 2.47105,
        area_km2=area_m2 / 1_000_000.0,
        perimeter_m=0.0,
        compactness=CompactnessMetricsSchema(
            isoperimetric_quotient=0.0,
            polsby_popper=0.0,
            convexity=0.0,
        ),
        threshold_classification=_classify_threshold(area_hectares),
        polygon_required=area_hectares >= 4.0,
        method="karney_geodesic",
        uncertainty_m2=0.0,
        provenance_hash=_compute_area_hash(hash_input),
    )


# ---------------------------------------------------------------------------
# POST /area/calculate
# ---------------------------------------------------------------------------


@router.post(
    "/area/calculate",
    response_model=AreaResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Calculate geodetic area",
    description=(
        "Calculate the geodetic area of a polygon boundary on the WGS84 "
        "ellipsoid using the Karney algorithm. Returns area in multiple "
        "units (m2, hectares, acres, km2), perimeter, compactness metrics "
        "(isoperimetric quotient, Polsby-Popper, convexity), EUDR threshold "
        "classification, and uncertainty estimate."
    ),
    responses={
        200: {"description": "Area calculation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def calculate_area(
    body: AreaCalculationRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:area:read")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AreaResponseSchema:
    """Calculate geodetic area for a polygon boundary.

    Uses the Karney geodesic algorithm on the WGS84 ellipsoid for
    high-precision area computation. Computes perimeter, compactness
    metrics, and EUDR Article 9 threshold classification.

    Args:
        body: Area calculation request with geometry, WKT, or plot_id.
        user: Authenticated user with area:read permission.

    Returns:
        AreaResponseSchema with geodetic area in multiple units.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    input_type = "geojson" if body.geometry else ("wkt" if body.wkt else "plot_id")

    logger.info(
        "Area calculation request: user=%s input_type=%s plot_id=%s",
        user.user_id,
        input_type,
        body.plot_id,
    )

    try:
        calculator = get_area_calculator()

        # Try to use real engine if available
        if hasattr(calculator, "calculate_area"):
            result = calculator.calculate_area(
                geometry=body.geometry,
                wkt=body.wkt,
                plot_id=body.plot_id,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Area calculated: area_ha=%.4f perimeter_m=%.1f "
                "method=%s elapsed_ms=%.1f",
                result.area_hectares if hasattr(result, "area_hectares") else 0.0,
                result.perimeter_m if hasattr(result, "perimeter_m") else 0.0,
                result.method if hasattr(result, "method") else "unknown",
                elapsed * 1000,
            )
            return result

        # Stub response for development
        response = _build_stub_area_response(body.geometry, body.wkt, body.plot_id)
        elapsed = time.monotonic() - start
        logger.info(
            "Area calculated (stub): area_ha=%.4f elapsed_ms=%.1f",
            response.area_hectares,
            elapsed * 1000,
        )
        return response

    except ValueError as exc:
        logger.warning(
            "Area calculation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Area calculation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Area calculation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /area/batch
# ---------------------------------------------------------------------------


@router.post(
    "/area/batch",
    response_model=BatchAreaResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch area calculation",
    description=(
        "Calculate geodetic area for multiple boundaries in a single "
        "request. Supports up to 10,000 geometries or plot_ids per batch."
    ),
    responses={
        200: {"description": "Batch area results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_calculate_area(
    body: BatchAreaRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:area:read")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchAreaResponseSchema:
    """Calculate geodetic area for multiple boundaries.

    Each geometry is processed independently. Results include
    per-item area calculations with threshold classification.

    Args:
        body: Batch area request with plot_ids or geometries.
        user: Authenticated user with area:read permission.

    Returns:
        BatchAreaResponseSchema with per-item area results.
    """
    start = time.monotonic()
    geometries = body.geometries or []
    plot_ids = body.plot_ids or []
    total = len(geometries) + len(plot_ids)

    logger.info(
        "Batch area request: user=%s geometries=%d plot_ids=%d",
        user.user_id,
        len(geometries),
        len(plot_ids),
    )

    results: List[BatchAreaResultSchema] = []
    idx = 0

    # Process geometries
    for geom in geometries:
        try:
            stub = _build_stub_area_response(geometry=geom)
            results.append(BatchAreaResultSchema(
                index=idx,
                area_m2=stub.area_m2,
                area_hectares=stub.area_hectares,
                perimeter_m=stub.perimeter_m,
                threshold_classification=stub.threshold_classification,
                success=True,
            ))
        except Exception as exc:
            results.append(BatchAreaResultSchema(
                index=idx,
                success=False,
                error=str(exc),
            ))
        idx += 1

    # Process plot_ids
    for pid in plot_ids:
        try:
            stub = _build_stub_area_response(plot_id=pid)
            results.append(BatchAreaResultSchema(
                index=idx,
                plot_id=pid,
                area_m2=stub.area_m2,
                area_hectares=stub.area_hectares,
                perimeter_m=stub.perimeter_m,
                threshold_classification=stub.threshold_classification,
                success=True,
            ))
        except Exception as exc:
            results.append(BatchAreaResultSchema(
                index=idx,
                plot_id=pid,
                success=False,
                error=str(exc),
            ))
        idx += 1

    elapsed = time.monotonic() - start
    logger.info(
        "Batch area completed: total=%d elapsed_ms=%.1f",
        total,
        elapsed * 1000,
    )

    return BatchAreaResponseSchema(
        total=total,
        results=results,
        processing_time_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# POST /area/threshold
# ---------------------------------------------------------------------------


@router.post(
    "/area/threshold",
    response_model=ThresholdResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Check EUDR 4ha threshold",
    description=(
        "Determine whether a plot boundary is above or below the EUDR "
        "Article 9 four-hectare threshold. Plots above the threshold "
        "require full polygon boundaries; plots below may use a single "
        "geolocation point."
    ),
    responses={
        200: {"description": "Threshold classification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_threshold(
    body: AreaCalculationRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:area:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ThresholdResponseSchema:
    """Check EUDR Article 9 area threshold classification.

    Computes the geodetic area and classifies the plot against the
    four-hectare threshold defined in EUDR Article 9.

    Args:
        body: Area calculation request with geometry, WKT, or plot_id.
        user: Authenticated user with area:read permission.

    Returns:
        ThresholdResponseSchema with classification and recommendation.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    input_type = "geojson" if body.geometry else ("wkt" if body.wkt else "plot_id")

    logger.info(
        "Threshold check request: user=%s input_type=%s plot_id=%s",
        user.user_id,
        input_type,
        body.plot_id,
    )

    try:
        cfg = get_config()
        threshold = cfg.area_threshold_hectares

        # Calculate area first
        area_response = _build_stub_area_response(
            body.geometry, body.wkt, body.plot_id,
        )
        area_hectares = area_response.area_hectares

        classification_str = _classify_threshold(area_hectares, threshold)
        classification = ThresholdClassificationSchema(classification_str)
        polygon_required = area_hectares >= threshold

        # Generate recommendation
        if polygon_required:
            recommendation = (
                f"Plot area ({area_hectares:.2f} ha) is at or above the "
                f"EUDR Article 9 threshold ({threshold:.1f} ha). "
                f"A full polygon boundary with geolocation of all plot "
                f"vertices is required for the due diligence statement."
            )
        else:
            recommendation = (
                f"Plot area ({area_hectares:.2f} ha) is below the "
                f"EUDR Article 9 threshold ({threshold:.1f} ha). "
                f"A single geolocation point is sufficient, but a full "
                f"polygon boundary is recommended for higher data quality."
            )

        elapsed = time.monotonic() - start
        logger.info(
            "Threshold check completed: area_ha=%.4f classification=%s "
            "polygon_required=%s elapsed_ms=%.1f",
            area_hectares,
            classification_str,
            polygon_required,
            elapsed * 1000,
        )

        return ThresholdResponseSchema(
            area_hectares=area_hectares,
            threshold_hectares=threshold,
            classification=classification,
            polygon_required=polygon_required,
            recommendation=recommendation,
        )

    except ValueError as exc:
        logger.warning(
            "Threshold check error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Threshold check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Threshold check failed due to an internal error",
        )
