# -*- coding: utf-8 -*-
"""
Coordinate Parsing Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for parsing raw coordinate strings into normalized decimal degrees,
detecting coordinate formats, and normalizing coordinates to WGS84.

Endpoints:
    POST /parse              - Parse a single raw coordinate string
    POST /parse/batch        - Parse multiple raw coordinate strings
    POST /parse/detect       - Detect coordinate format without parsing
    POST /parse/normalize    - Parse and normalize to WGS84

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import logging
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
    BatchParseRequestSchema,
    BatchParseResponseSchema,
    FormatDetectionResponseSchema,
    NormalizeResponseSchema,
    ParseRequestSchema,
    ParseResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Coordinate Parsing"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /parse
# ---------------------------------------------------------------------------


@router.post(
    "/parse",
    response_model=ParseResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Parse a raw coordinate string",
    description=(
        "Parse a single raw GPS coordinate string into decimal degrees. "
        "Supports DD (5.6037, -0.1870), DMS (5d36'13.4\"N 0d11'13.1\"W), "
        "DDM (5d36.2233'N 0d11.2183'W), UTM (30N 808820 620350), "
        "and MGRS (30NUN0882020350) formats. Format is automatically "
        "detected unless a format_hint is provided."
    ),
    responses={
        200: {"description": "Successfully parsed coordinate"},
        400: {"model": ErrorResponse, "description": "Parse error or invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def parse_coordinate(
    body: ParseRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:parse:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ParseResponseSchema:
    """Parse a raw coordinate string into decimal degrees.

    Automatically detects the coordinate format (DD, DMS, DDM, UTM, MGRS)
    and converts to decimal degrees in the source datum. Optionally
    applies datum context for subsequent transformation.

    Args:
        body: Parse request with raw coordinate string.
        request: FastAPI request object.
        user: Authenticated user with parse:write permission.

    Returns:
        ParseResponseSchema with parsed lat/lon and detection metadata.

    Raises:
        HTTPException: 400 if parsing fails, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Parse request: user=%s input='%s' format_hint=%s datum=%s",
        user.user_id,
        body.input[:80],
        body.format_hint,
        body.datum,
    )

    try:
        service = get_gps_validator_service()

        result = service.parse_coordinate(
            raw_input=body.input,
            format_hint=body.format_hint,
            datum=body.datum,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"{body.input}|{result.get('latitude', 0)}|{result.get('longitude', 0)}"
        )

        logger.info(
            "Parse completed: user=%s format=%s confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            result.get("detected_format", "unknown"),
            result.get("confidence", 0.0),
            elapsed * 1000,
        )

        return ParseResponseSchema(
            latitude=result["latitude"],
            longitude=result["longitude"],
            detected_format=result.get("detected_format", "unknown"),
            confidence=result.get("confidence", 0.0),
            datum=result.get("datum", body.datum or "wgs84"),
            warnings=result.get("warnings", []),
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Parse error: user=%s input='%s' error=%s",
            user.user_id,
            body.input[:80],
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse coordinate: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Parse failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coordinate parsing failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /parse/batch
# ---------------------------------------------------------------------------


@router.post(
    "/parse/batch",
    response_model=BatchParseResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Parse multiple raw coordinate strings",
    description=(
        "Parse multiple raw GPS coordinate strings in a single batch request. "
        "Each coordinate is independently parsed and format-detected. "
        "Returns per-coordinate results with any errors indexed to the "
        "original input position. Maximum 10,000 coordinates per batch."
    ),
    responses={
        200: {"description": "Batch parse results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def parse_coordinates_batch(
    body: BatchParseRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:parse:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchParseResponseSchema:
    """Parse multiple raw coordinate strings in batch.

    Each coordinate is parsed independently. Errors are captured per-index
    without failing the entire batch.

    Args:
        body: Batch parse request with list of raw coordinates.
        request: FastAPI request object.
        user: Authenticated user with parse:write permission.

    Returns:
        BatchParseResponseSchema with per-coordinate results and errors.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Batch parse request: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        results: List[ParseResponseSchema] = []
        errors: List[Dict[str, Any]] = []
        successful = 0

        for idx, coord in enumerate(body.coordinates):
            try:
                result = service.parse_coordinate(
                    raw_input=coord.input,
                    format_hint=coord.format_hint,
                    datum=coord.datum,
                )
                provenance = _compute_provenance(
                    f"{coord.input}|{result.get('latitude', 0)}|"
                    f"{result.get('longitude', 0)}"
                )
                results.append(ParseResponseSchema(
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    detected_format=result.get("detected_format", "unknown"),
                    confidence=result.get("confidence", 0.0),
                    datum=result.get("datum", coord.datum or "wgs84"),
                    warnings=result.get("warnings", []),
                    provenance_hash=provenance,
                ))
                successful += 1
            except (ValueError, KeyError) as exc:
                errors.append({
                    "index": idx,
                    "input": coord.input[:100],
                    "error": str(exc),
                })

        elapsed = time.monotonic() - start
        batch_provenance = _compute_provenance(
            f"batch|{total}|{successful}|{len(errors)}"
        )

        logger.info(
            "Batch parse completed: user=%s total=%d successful=%d "
            "failed=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            len(errors),
            elapsed * 1000,
        )

        return BatchParseResponseSchema(
            total=total,
            successful=successful,
            failed=len(errors),
            results=results,
            errors=errors,
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
            "Batch parse failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch coordinate parsing failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /parse/detect
# ---------------------------------------------------------------------------


@router.post(
    "/parse/detect",
    response_model=FormatDetectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Detect coordinate format without parsing",
    description=(
        "Analyze a raw coordinate string to detect its format without "
        "performing full parsing. Returns the most likely format with "
        "confidence score and alternative format candidates."
    ),
    responses={
        200: {"description": "Format detection result"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_format(
    body: ParseRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:parse:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FormatDetectionResponseSchema:
    """Detect the format of a raw coordinate string.

    Analyzes the input string to determine whether it is DD, DMS, DDM,
    UTM, MGRS, or another supported format. Returns confidence scores
    for the primary detection and alternative candidates.

    Args:
        body: Request with raw coordinate string.
        request: FastAPI request object.
        user: Authenticated user with parse:read permission.

    Returns:
        FormatDetectionResponseSchema with format and alternatives.

    Raises:
        HTTPException: 400 if input cannot be analyzed, 500 on error.
    """
    start = time.monotonic()
    logger.info(
        "Format detection request: user=%s input='%s'",
        user.user_id,
        body.input[:80],
    )

    try:
        service = get_gps_validator_service()

        result = service.detect_format(raw_input=body.input)

        elapsed = time.monotonic() - start
        logger.info(
            "Format detection completed: user=%s format=%s confidence=%.2f "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("detected_format", "unknown"),
            result.get("confidence", 0.0),
            elapsed * 1000,
        )

        return FormatDetectionResponseSchema(
            detected_format=result.get("detected_format", "unknown"),
            confidence=result.get("confidence", 0.0),
            alternatives=result.get("alternatives", []),
            input_analyzed=body.input[:200],
        )

    except ValueError as exc:
        logger.warning(
            "Format detection error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format detection failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Format detection failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Format detection failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /parse/normalize
# ---------------------------------------------------------------------------


@router.post(
    "/parse/normalize",
    response_model=NormalizeResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Parse and normalize coordinate to WGS84",
    description=(
        "Parse a raw coordinate string and normalize it to WGS84 decimal "
        "degrees. Performs format detection, parsing, and optional datum "
        "transformation in a single operation. Returns the WGS84 "
        "coordinate with displacement information if datum transformation "
        "was applied."
    ),
    responses={
        200: {"description": "Normalized WGS84 coordinate"},
        400: {"model": ErrorResponse, "description": "Parse or normalization error"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def normalize_coordinate(
    body: ParseRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:parse:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> NormalizeResponseSchema:
    """Parse and normalize a coordinate to WGS84 decimal degrees.

    Combines format detection, parsing, and datum transformation into
    a single pipeline. If the source datum is not WGS84, a Helmert
    7-parameter transformation is applied.

    Args:
        body: Request with raw coordinate string and optional datum.
        request: FastAPI request object.
        user: Authenticated user with parse:write permission.

    Returns:
        NormalizeResponseSchema with WGS84 coordinate and metadata.

    Raises:
        HTTPException: 400 if normalization fails, 500 on error.
    """
    start = time.monotonic()
    logger.info(
        "Normalize request: user=%s input='%s' datum=%s",
        user.user_id,
        body.input[:80],
        body.datum,
    )

    try:
        service = get_gps_validator_service()

        result = service.normalize_coordinate(
            raw_input=body.input,
            format_hint=body.format_hint,
            source_datum=body.datum,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"normalize|{body.input}|{result.get('latitude', 0)}|"
            f"{result.get('longitude', 0)}"
        )

        logger.info(
            "Normalize completed: user=%s displacement_m=%.3f "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("displacement_m", 0.0),
            elapsed * 1000,
        )

        return NormalizeResponseSchema(
            latitude=result["latitude"],
            longitude=result["longitude"],
            original_input=body.input[:200],
            source_datum=result.get("source_datum", body.datum or "wgs84"),
            target_datum="wgs84",
            displacement_m=result.get("displacement_m", 0.0),
            format_detected=result.get("format_detected", "unknown"),
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Normalize error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Coordinate normalization failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Normalize failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coordinate normalization failed due to an internal error",
        )
