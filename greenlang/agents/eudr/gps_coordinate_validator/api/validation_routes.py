# -*- coding: utf-8 -*-
"""
Coordinate Validation Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for validating coordinate pairs against WGS84 range constraints,
detecting lat/lon swaps, checking for null island, and identifying
duplicates within coordinate sets.

Endpoints:
    POST /validate           - Validate a single coordinate pair
    POST /validate/batch     - Validate multiple coordinate pairs
    POST /validate/range     - Quick range check (bounds, NaN, null island)
    POST /validate/swap      - Detect swapped latitude/longitude
    POST /validate/duplicates - Detect duplicates in a coordinate set

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.gps_coordinate_validator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_gps_validator_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    BatchValidateRequestSchema,
    BatchValidateResponseSchema,
    CoordinatePairSchema,
    DuplicateDetectionRequestSchema,
    DuplicateDetectionResponseSchema,
    RangeCheckResponseSchema,
    SwapDetectionRequestSchema,
    SwapDetectionResponseSchema,
    ValidateRequestSchema,
    ValidationErrorSchema,
    ValidationResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Coordinate Validation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=ValidationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Validate a single coordinate pair",
    description=(
        "Validate a GPS coordinate pair (WGS84) for range compliance, "
        "null island detection, sign errors, swap detection, precision "
        "adequacy, and boundary value suspicion. Returns validation "
        "results with errors, warnings, and auto-correction suggestions."
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
    body: ValidateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:validate:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ValidationResponseSchema:
    """Validate a single coordinate pair.

    Runs all validation checks including range, null island, swap
    detection, sign error, and boundary value analysis. Returns
    a comprehensive validation result with optional auto-corrections.

    Args:
        body: Validate request with latitude, longitude, and context.
        request: FastAPI request object.
        user: Authenticated user with validate:write permission.

    Returns:
        ValidationResponseSchema with errors, warnings, and corrections.

    Raises:
        HTTPException: 400 if input invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Validation request: user=%s lat=%.6f lon=%.6f country=%s commodity=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.country_iso,
        body.commodity,
    )

    try:
        service = get_gps_validator_service()

        result = service.validate_coordinate(
            latitude=body.latitude,
            longitude=body.longitude,
            country_iso=body.country_iso,
            commodity=body.commodity,
            source_type=body.source_type,
            altitude=body.altitude,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"validate|{body.latitude}|{body.longitude}|{result.get('is_valid', False)}"
        )

        errors = [
            ValidationErrorSchema(**e) for e in result.get("errors", [])
        ]
        normalized = CoordinatePairSchema(
            latitude=result.get("normalized_lat", body.latitude),
            longitude=result.get("normalized_lon", body.longitude),
            altitude=body.altitude,
            datum="WGS84",
            commodity=body.commodity,
            country_iso=body.country_iso,
            source_type=body.source_type or "unknown",
        )

        logger.info(
            "Validation completed: user=%s is_valid=%s errors=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("is_valid", False),
            len(errors),
            elapsed * 1000,
        )

        return ValidationResponseSchema(
            is_valid=result.get("is_valid", True),
            errors=errors,
            warnings=result.get("warnings", []),
            auto_corrections=result.get("auto_corrections", []),
            normalized=normalized,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Validation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coordinate validation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /validate/batch
# ---------------------------------------------------------------------------


@router.post(
    "/validate/batch",
    response_model=BatchValidateResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Validate multiple coordinate pairs",
    description=(
        "Validate multiple GPS coordinate pairs in a single batch request. "
        "Each coordinate runs through all validation checks independently. "
        "Returns aggregate counts and per-coordinate results. "
        "Maximum 10,000 coordinates per batch."
    ),
    responses={
        200: {"description": "Batch validation results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_coordinates_batch(
    body: BatchValidateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:validate:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchValidateResponseSchema:
    """Validate multiple coordinate pairs in batch.

    Performs per-coordinate validation with aggregate statistics.
    Each coordinate is validated independently so individual failures
    do not affect other coordinates in the batch.

    Args:
        body: Batch validate request with list of coordinates.
        request: FastAPI request object.
        user: Authenticated user with validate:write permission.

    Returns:
        BatchValidateResponseSchema with results and summary counts.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Batch validation request: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        results: List[ValidationResponseSchema] = []
        valid_count = 0
        invalid_count = 0
        warning_count = 0
        auto_corrected_count = 0

        for coord in body.coordinates:
            try:
                result = service.validate_coordinate(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    country_iso=coord.country_iso,
                    commodity=coord.commodity,
                    source_type=coord.source_type,
                    altitude=coord.altitude,
                )

                errors = [
                    ValidationErrorSchema(**e) for e in result.get("errors", [])
                ]
                normalized = CoordinatePairSchema(
                    latitude=result.get("normalized_lat", coord.latitude),
                    longitude=result.get("normalized_lon", coord.longitude),
                    altitude=coord.altitude,
                    datum="WGS84",
                    commodity=coord.commodity,
                    country_iso=coord.country_iso,
                    source_type=coord.source_type,
                )

                is_valid = result.get("is_valid", True)
                has_warnings = len(result.get("warnings", [])) > 0
                has_corrections = len(result.get("auto_corrections", [])) > 0

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                if has_warnings:
                    warning_count += 1
                if has_corrections:
                    auto_corrected_count += 1

                provenance = _compute_provenance(
                    f"validate|{coord.latitude}|{coord.longitude}|{is_valid}"
                )

                results.append(ValidationResponseSchema(
                    is_valid=is_valid,
                    errors=errors,
                    warnings=result.get("warnings", []),
                    auto_corrections=result.get("auto_corrections", []),
                    normalized=normalized,
                    provenance_hash=provenance,
                ))

            except (ValueError, KeyError) as exc:
                invalid_count += 1
                normalized_fallback = CoordinatePairSchema(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    datum="WGS84",
                    source_type=coord.source_type,
                )
                results.append(ValidationResponseSchema(
                    is_valid=False,
                    errors=[ValidationErrorSchema(
                        error_type="processing_error",
                        description=str(exc),
                        severity="error",
                        auto_correctable=False,
                    )],
                    warnings=[],
                    auto_corrections=[],
                    normalized=normalized_fallback,
                ))

        elapsed = time.monotonic() - start
        batch_provenance = _compute_provenance(
            f"batch_validate|{total}|{valid_count}|{invalid_count}"
        )

        logger.info(
            "Batch validation completed: user=%s total=%d valid=%d "
            "invalid=%d warnings=%d auto_corrected=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            valid_count,
            invalid_count,
            warning_count,
            auto_corrected_count,
            elapsed * 1000,
        )

        return BatchValidateResponseSchema(
            total=total,
            valid=valid_count,
            invalid=invalid_count,
            warnings=warning_count,
            auto_corrected=auto_corrected_count,
            results=results,
            processing_time_ms=elapsed * 1000,
            provenance_hash=batch_provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch validation failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch validation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /validate/range
# ---------------------------------------------------------------------------


@router.post(
    "/validate/range",
    response_model=RangeCheckResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Quick coordinate range check",
    description=(
        "Perform a lightweight range check on a coordinate pair. "
        "Validates WGS84 bounds (-90/+90 lat, -180/+180 lon), "
        "checks for null island (0, 0), NaN/infinity values, "
        "and boundary value suspicion (exactly 0, 90, 180, etc)."
    ),
    responses={
        200: {"description": "Range check result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def range_check(
    body: ValidateRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:validate:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RangeCheckResponseSchema:
    """Quick range check for a coordinate pair.

    Performs lightweight validation without full plausibility analysis.
    Checks WGS84 bounds, NaN, infinity, null island, and suspicious
    boundary values.

    Args:
        body: Validate request with latitude and longitude.
        request: FastAPI request object.
        user: Authenticated user with validate:read permission.

    Returns:
        RangeCheckResponseSchema with check results.
    """
    logger.info(
        "Range check: user=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    lat = body.latitude
    lon = body.longitude
    details: List[str] = []

    is_nan = math.isnan(lat) or math.isnan(lon)
    lat_in_range = -90.0 <= lat <= 90.0 if not is_nan else False
    lon_in_range = -180.0 <= lon <= 180.0 if not is_nan else False

    # Null island: within 0.001 degrees of (0, 0)
    is_null_island = False
    if not is_nan and abs(lat) < 0.001 and abs(lon) < 0.001:
        is_null_island = True
        details.append(
            "Coordinate is at or near null island (0, 0). "
            "This is rarely a valid production plot location."
        )

    # Boundary value detection
    is_boundary = False
    boundary_values = {0.0, 90.0, -90.0, 180.0, -180.0, 45.0, -45.0}
    if not is_nan:
        if lat in boundary_values or lon in boundary_values:
            is_boundary = True
            details.append(
                "Coordinate contains a suspicious boundary value. "
                "This may indicate placeholder or default data."
            )

    if is_nan:
        details.append("Coordinate contains NaN value(s).")
    if not lat_in_range and not is_nan:
        details.append(f"Latitude {lat} is out of WGS84 range [-90, 90].")
    if not lon_in_range and not is_nan:
        details.append(f"Longitude {lon} is out of WGS84 range [-180, 180].")

    return RangeCheckResponseSchema(
        latitude=lat if not is_nan else 0.0,
        longitude=lon if not is_nan else 0.0,
        latitude_in_range=lat_in_range,
        longitude_in_range=lon_in_range,
        is_null_island=is_null_island,
        is_nan=is_nan,
        is_boundary=is_boundary,
        details=details,
    )


# ---------------------------------------------------------------------------
# POST /validate/swap
# ---------------------------------------------------------------------------


@router.post(
    "/validate/swap",
    response_model=SwapDetectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Detect swapped latitude/longitude",
    description=(
        "Analyze a coordinate pair to determine if latitude and longitude "
        "values appear to be swapped. Uses multiple heuristics including "
        "range analysis, country boundary matching, and hemisphere "
        "consistency to assess swap probability."
    ),
    responses={
        200: {"description": "Swap detection result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_swap(
    body: SwapDetectionRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:validate:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SwapDetectionResponseSchema:
    """Detect if latitude and longitude are swapped.

    Uses range analysis and optional country boundary matching to
    determine swap probability. If a swap is detected with high
    confidence, provides the corrected coordinate.

    Args:
        body: Request with latitude, longitude, and optional country.
        request: FastAPI request object.
        user: Authenticated user with validate:read permission.

    Returns:
        SwapDetectionResponseSchema with detection result.

    Raises:
        HTTPException: 400 if input invalid, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Swap detection: user=%s lat=%.6f lon=%.6f country=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.country_iso,
    )

    try:
        service = get_gps_validator_service()

        result = service.detect_swap(
            latitude=body.latitude,
            longitude=body.longitude,
            country_iso=body.country_iso,
        )

        elapsed = time.monotonic() - start
        corrected = None
        if result.get("is_swapped", False):
            corrected = CoordinatePairSchema(
                latitude=body.longitude,
                longitude=body.latitude,
                datum="WGS84",
            )

        logger.info(
            "Swap detection completed: user=%s is_swapped=%s "
            "confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            result.get("is_swapped", False),
            result.get("confidence", 0.0),
            elapsed * 1000,
        )

        return SwapDetectionResponseSchema(
            is_swapped=result.get("is_swapped", False),
            confidence=result.get("confidence", 0.0),
            corrected=corrected,
            reasoning=result.get("reasoning", ""),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Swap detection failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Swap detection failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /validate/duplicates
# ---------------------------------------------------------------------------


@router.post(
    "/validate/duplicates",
    response_model=DuplicateDetectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Detect duplicate coordinates in a set",
    description=(
        "Analyze a set of coordinates to detect exact duplicates and "
        "near-duplicates within a configurable distance threshold. "
        "Returns pairs of duplicate indices with their distances."
    ),
    responses={
        200: {"description": "Duplicate detection results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_duplicates(
    body: DuplicateDetectionRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:validate:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DuplicateDetectionResponseSchema:
    """Detect duplicates and near-duplicates in a coordinate set.

    Compares all coordinate pairs to find exact matches and coordinates
    within the specified distance threshold (in metres). Uses the
    Haversine formula for distance calculation.

    Args:
        body: Request with coordinates list and threshold.
        request: FastAPI request object.
        user: Authenticated user with validate:read permission.

    Returns:
        DuplicateDetectionResponseSchema with duplicate pairs.

    Raises:
        HTTPException: 400 if input invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Duplicate detection: user=%s total=%d threshold_m=%.1f",
        user.user_id,
        total,
        body.threshold_m,
    )

    try:
        service = get_gps_validator_service()

        coords_data = [
            {
                "latitude": c.latitude,
                "longitude": c.longitude,
            }
            for c in body.coordinates
        ]

        result = service.detect_duplicates(
            coordinates=coords_data,
            threshold_m=body.threshold_m,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Duplicate detection completed: user=%s total=%d "
            "exact=%d near=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            result.get("exact_duplicates", 0),
            result.get("near_duplicates", 0),
            elapsed * 1000,
        )

        return DuplicateDetectionResponseSchema(
            total_coordinates=total,
            exact_duplicates=result.get("exact_duplicates", 0),
            near_duplicates=result.get("near_duplicates", 0),
            duplicate_pairs=result.get("duplicate_pairs", []),
            near_duplicate_pairs=result.get("near_duplicate_pairs", []),
            threshold_m=body.threshold_m,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Duplicate detection failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Duplicate detection failed due to an internal error",
        )
