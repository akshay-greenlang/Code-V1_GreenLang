# -*- coding: utf-8 -*-
"""
Coordinate Validation Routes - AGENT-EUDR-002 Geolocation Verification API

Endpoints for validating single and batch coordinate pairs against WGS84
bounds, country boundaries, precision requirements, and anomaly detection.

Endpoints:
    POST /coordinates       - Validate a single coordinate pair
    POST /coordinates/batch - Validate multiple coordinate pairs in batch

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.geolocation_verification.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_coordinate_validator,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    BatchCoordinateRequest,
    BatchCoordinateResponse,
    CoordinateValidationRequest,
    CoordinateValidationResponse,
)
from greenlang.agents.eudr.geolocation_verification.models import (
    VerifyCoordinateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Coordinate Validation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_response(result) -> CoordinateValidationResponse:
    """Convert a CoordinateValidationResult dataclass to API response model.

    Args:
        result: CoordinateValidationResult from the validation engine.

    Returns:
        CoordinateValidationResponse Pydantic model for serialization.
    """
    return CoordinateValidationResponse(
        validation_id=result.validation_id,
        lat=result.lat,
        lon=result.lon,
        is_valid=result.is_valid,
        wgs84_valid=result.wgs84_valid,
        precision_decimal_places=result.precision_decimal_places,
        precision_score=result.precision_score,
        transposition_detected=result.transposition_detected,
        country_match=result.country_match,
        resolved_country=result.resolved_country,
        is_on_land=result.is_on_land,
        is_duplicate=result.is_duplicate,
        elevation_m=result.elevation_m,
        elevation_plausible=result.elevation_plausible,
        cluster_anomaly=result.cluster_anomaly,
        issues=[i.to_dict() for i in result.issues],
        provenance_hash=result.provenance_hash,
        validated_at=result.validated_at,
    )


# ---------------------------------------------------------------------------
# POST /coordinates
# ---------------------------------------------------------------------------


@router.post(
    "/coordinates",
    response_model=CoordinateValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a single coordinate pair",
    description=(
        "Validate a single GPS coordinate pair (WGS84) against bounds, "
        "precision requirements, country boundary matching, land/ocean "
        "detection, elevation plausibility, and transposition detection. "
        "Returns detailed validation results with a list of issues."
    ),
    responses={
        200: {"description": "Coordinate validation result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_coordinate(
    body: CoordinateValidationRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:coordinates:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CoordinateValidationResponse:
    """Validate a single coordinate pair.

    Performs WGS84 bounds validation, precision assessment, country
    boundary matching, land/ocean detection, elevation plausibility,
    and transposition detection.

    Args:
        body: Coordinate validation request with lat, lon, country, commodity.
        user: Authenticated user with coordinates:write permission.

    Returns:
        CoordinateValidationResponse with full validation results.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Coordinate validation request: user=%s lat=%.6f lon=%.6f country=%s",
        user.user_id,
        body.lat,
        body.lon,
        body.declared_country_code,
    )

    try:
        validator = get_coordinate_validator()

        coordinate_input = VerifyCoordinateRequest(
            lat=body.lat,
            lon=body.lon,
            declared_country=body.declared_country_code,
            commodity=body.commodity,
        )

        result = validator.validate(coordinate_input)

        elapsed = time.monotonic() - start
        logger.info(
            "Coordinate validation completed: validation_id=%s is_valid=%s "
            "precision=%d elapsed_ms=%.1f",
            result.validation_id,
            result.is_valid,
            result.precision_decimal_places,
            elapsed * 1000,
        )

        return _result_to_response(result)

    except ValueError as exc:
        logger.warning(
            "Coordinate validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Coordinate validation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coordinate validation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /coordinates/batch
# ---------------------------------------------------------------------------


@router.post(
    "/coordinates/batch",
    response_model=BatchCoordinateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate multiple coordinate pairs in batch",
    description=(
        "Validate multiple GPS coordinate pairs in a single request. "
        "Performs all single-coordinate checks plus batch-level duplicate "
        "detection and cluster anomaly analysis. Maximum 10,000 coordinates "
        "per batch."
    ),
    responses={
        200: {"description": "Batch coordinate validation results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_coordinates_batch(
    body: BatchCoordinateRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:coordinates:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BatchCoordinateResponse:
    """Validate multiple coordinate pairs in a single batch request.

    Performs per-coordinate validation plus batch-level duplicate detection
    and cluster anomaly analysis.

    Args:
        body: Batch coordinate request with list of coordinates.
        user: Authenticated user with coordinates:write permission.

    Returns:
        BatchCoordinateResponse with per-coordinate results and batch summary.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    total = len(body.coordinates)

    logger.info(
        "Batch coordinate validation: user=%s batch_id=%s total=%d",
        user.user_id,
        batch_id,
        total,
    )

    try:
        validator = get_coordinate_validator()

        coordinate_inputs = [
            VerifyCoordinateRequest(
                lat=c.lat,
                lon=c.lon,
                declared_country=c.declared_country_code,
                commodity=c.commodity,
            )
            for c in body.coordinates
        ]

        results = validator.validate_batch(coordinate_inputs)

        responses: List[CoordinateValidationResponse] = []
        valid_count = 0
        invalid_count = 0

        for result in results:
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            responses.append(_result_to_response(result))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch coordinate validation completed: batch_id=%s "
            "total=%d valid=%d invalid=%d elapsed_ms=%.1f",
            batch_id,
            total,
            valid_count,
            invalid_count,
            elapsed * 1000,
        )

        return BatchCoordinateResponse(
            batch_id=batch_id,
            total_coordinates=total,
            valid_count=valid_count,
            invalid_count=invalid_count,
            results=responses,
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Batch coordinate validation error: user=%s batch_id=%s error=%s",
            user.user_id,
            batch_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch coordinate validation failed: user=%s batch_id=%s error=%s",
            user.user_id,
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch coordinate validation failed due to an internal error",
        )
