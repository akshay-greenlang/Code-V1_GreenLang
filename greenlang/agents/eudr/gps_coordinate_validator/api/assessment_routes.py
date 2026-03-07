# -*- coding: utf-8 -*-
"""
Accuracy Assessment Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for computing accuracy scores, quality tier classifications,
precision analysis, and comprehensive coordinate assessments combining
validation, plausibility, and precision into a single pipeline.

Endpoints:
    POST /assess              - Full accuracy assessment
    POST /assess/batch        - Batch accuracy assessment
    GET  /assess/{id}         - Retrieve assessment result by ID
    POST /assess/precision    - Precision analysis only

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

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.gps_coordinate_validator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_gps_validator_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    AccuracyScoreSchema,
    AssessmentRequestSchema,
    AssessmentResponseSchema,
    BatchAssessmentRequestSchema,
    BatchAssessmentResponseSchema,
    BatchSummaryResponseSchema,
    CoordinatePairSchema,
    PlausibilityResponseSchema,
    PrecisionRequestSchema,
    PrecisionResponseSchema,
    ValidationErrorSchema,
    ValidationResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Accuracy Assessment"])


# ---------------------------------------------------------------------------
# In-memory assessment store (replaced by database in production)
# ---------------------------------------------------------------------------

_assessment_store: Dict[str, Dict[str, Any]] = {}


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


def _build_assessment_response(
    assessment_id: str,
    coord: AssessmentRequestSchema,
    result: Dict[str, Any],
) -> AssessmentResponseSchema:
    """Build an AssessmentResponseSchema from service result.

    Args:
        assessment_id: Unique assessment identifier.
        coord: Original assessment request.
        result: Service result dictionary.

    Returns:
        Fully populated AssessmentResponseSchema.
    """
    coordinate = CoordinatePairSchema(
        latitude=coord.latitude,
        longitude=coord.longitude,
        datum="WGS84",
        commodity=coord.commodity,
        country_iso=coord.country_iso,
        source_type=coord.source_type or "unknown",
    )

    accuracy = AccuracyScoreSchema(
        overall_score=result.get("overall_score", 0.0),
        tier=result.get("tier", "fail"),
        precision_score=result.get("precision_score", 0.0),
        plausibility_score=result.get("plausibility_score", 0.0),
        consistency_score=result.get("consistency_score", 0.0),
        source_score=result.get("source_score", 0.0),
        confidence_interval_m=result.get("confidence_interval_m", 0.0),
        explanations=result.get("explanations", []),
    )

    validation_data = result.get("validation", {})
    validation = ValidationResponseSchema(
        is_valid=validation_data.get("is_valid", True),
        errors=[
            ValidationErrorSchema(**e)
            for e in validation_data.get("errors", [])
        ],
        warnings=validation_data.get("warnings", []),
        auto_corrections=validation_data.get("auto_corrections", []),
        normalized=coordinate,
    )

    precision_data = result.get("precision", {})
    precision = PrecisionResponseSchema(
        decimal_places_lat=precision_data.get("decimal_places_lat", 0),
        decimal_places_lon=precision_data.get("decimal_places_lon", 0),
        ground_resolution_lat_m=precision_data.get("ground_resolution_lat_m", 0.0),
        ground_resolution_lon_m=precision_data.get("ground_resolution_lon_m", 0.0),
        level=precision_data.get("level", "inadequate"),
        eudr_adequate=precision_data.get("eudr_adequate", False),
        is_truncated=precision_data.get("is_truncated", False),
        is_rounded=precision_data.get("is_rounded", False),
    )

    plausibility_data = result.get("plausibility", {})
    plausibility = PlausibilityResponseSchema(
        is_on_land=plausibility_data.get("is_on_land", True),
        detected_country=plausibility_data.get("detected_country"),
        country_match=plausibility_data.get("country_match"),
        commodity_plausible=plausibility_data.get("commodity_plausible"),
        elevation_plausible=plausibility_data.get("elevation_plausible"),
        is_urban=plausibility_data.get("is_urban", False),
        is_protected=plausibility_data.get("is_protected", False),
        land_use=plausibility_data.get("land_use"),
        distance_to_coast_km=plausibility_data.get("distance_to_coast_km"),
        details=plausibility_data.get("details", {}),
    )

    provenance = _compute_provenance(
        f"assess|{assessment_id}|{coord.latitude}|{coord.longitude}|"
        f"{accuracy.overall_score}"
    )

    return AssessmentResponseSchema(
        assessment_id=assessment_id,
        coordinate=coordinate,
        accuracy=accuracy,
        validation=validation,
        precision=precision,
        plausibility=plausibility,
        provenance_hash=provenance,
    )


# ---------------------------------------------------------------------------
# POST /assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=AssessmentResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Full accuracy assessment",
    description=(
        "Perform a comprehensive accuracy assessment of a GPS coordinate "
        "combining validation checks, plausibility analysis, precision "
        "measurement, and source reliability scoring. Returns an overall "
        "accuracy score (0-100) with quality tier classification "
        "(gold/silver/bronze/fail) and detailed sub-score breakdown."
    ),
    responses={
        200: {"description": "Accuracy assessment result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_coordinate(
    body: AssessmentRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:assess:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AssessmentResponseSchema:
    """Perform full accuracy assessment on a coordinate.

    Combines validation, plausibility, precision analysis, and source
    scoring into a single composite accuracy score with tier classification.

    Args:
        body: Assessment request with coordinate and context.
        request: FastAPI request object.
        user: Authenticated user with assess:write permission.

    Returns:
        AssessmentResponseSchema with score breakdown and tier.

    Raises:
        HTTPException: 400 if input invalid, 500 on processing error.
    """
    start = time.monotonic()
    assessment_id = f"assess-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Assessment request: user=%s assessment_id=%s lat=%.6f lon=%.6f "
        "source=%s commodity=%s",
        user.user_id,
        assessment_id,
        body.latitude,
        body.longitude,
        body.source_type,
        body.commodity,
    )

    try:
        service = get_gps_validator_service()

        result = service.assess_coordinate(
            latitude=body.latitude,
            longitude=body.longitude,
            source_type=body.source_type,
            commodity=body.commodity,
            country_iso=body.country_iso,
        )

        response = _build_assessment_response(assessment_id, body, result)

        # Store for later retrieval
        _assessment_store[assessment_id] = {
            "response": response.model_dump(mode="json"),
            "user_id": user.user_id,
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Assessment completed: assessment_id=%s score=%.1f tier=%s "
            "elapsed_ms=%.1f",
            assessment_id,
            response.accuracy.overall_score,
            response.accuracy.tier,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Assessment error: user=%s error=%s",
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
            "Assessment failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Accuracy assessment failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /assess/batch
# ---------------------------------------------------------------------------


@router.post(
    "/assess/batch",
    response_model=BatchAssessmentResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch accuracy assessment",
    description=(
        "Perform accuracy assessments on multiple coordinates in a single "
        "request. Returns per-coordinate results and an aggregate summary "
        "with tier distribution. Maximum 5,000 coordinates per batch."
    ),
    responses={
        200: {"description": "Batch assessment results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_coordinates_batch(
    body: BatchAssessmentRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:assess:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchAssessmentResponseSchema:
    """Perform batch accuracy assessment on multiple coordinates.

    Each coordinate is assessed independently with aggregate statistics
    computed across the batch.

    Args:
        body: Batch assessment request with coordinates.
        request: FastAPI request object.
        user: Authenticated user with assess:write permission.

    Returns:
        BatchAssessmentResponseSchema with results and summary.

    Raises:
        HTTPException: 400 if request invalid, 500 on internal error.
    """
    start = time.monotonic()
    total = len(body.coordinates)

    logger.info(
        "Batch assessment request: user=%s total=%d",
        user.user_id,
        total,
    )

    try:
        service = get_gps_validator_service()

        results: List[AssessmentResponseSchema] = []
        valid_count = 0
        invalid_count = 0
        tier_counts: Dict[str, int] = {
            "gold": 0, "silver": 0, "bronze": 0, "fail": 0,
        }
        precision_counts: Dict[str, int] = {}
        error_counts: Dict[str, int] = {}

        for coord in body.coordinates:
            assessment_id = f"assess-{uuid.uuid4().hex[:12]}"
            try:
                result = service.assess_coordinate(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    source_type=coord.source_type,
                    commodity=coord.commodity,
                    country_iso=coord.country_iso,
                )

                response = _build_assessment_response(
                    assessment_id, coord, result
                )
                results.append(response)

                # Aggregate counts
                tier = response.accuracy.tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                level = response.precision.level
                precision_counts[level] = precision_counts.get(level, 0) + 1

                if response.validation.is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    for err in response.validation.errors:
                        et = err.error_type
                        error_counts[et] = error_counts.get(et, 0) + 1

            except (ValueError, KeyError):
                invalid_count += 1
                tier_counts["fail"] = tier_counts.get("fail", 0) + 1

        elapsed = time.monotonic() - start
        total_warnings = sum(
            len(r.validation.warnings) for r in results
        )

        summary = BatchSummaryResponseSchema(
            total=total,
            valid=valid_count,
            invalid=invalid_count,
            warning_count=total_warnings,
            error_breakdown=error_counts,
            precision_distribution=precision_counts,
            tier_distribution=tier_counts,
            recommendations=_generate_recommendations(
                tier_counts, error_counts, precision_counts
            ),
        )

        provenance = _compute_provenance(
            f"batch_assess|{total}|{valid_count}|{invalid_count}"
        )

        logger.info(
            "Batch assessment completed: user=%s total=%d valid=%d "
            "invalid=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            valid_count,
            invalid_count,
            elapsed * 1000,
        )

        return BatchAssessmentResponseSchema(
            total=total,
            results=results,
            summary=summary,
            processing_time_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch assessment failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch assessment failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /assess/{assessment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/assess/{assessment_id}",
    response_model=AssessmentResponseSchema,
    summary="Retrieve assessment result by ID",
    description=(
        "Retrieve a previously computed accuracy assessment by its "
        "unique identifier. Returns the full assessment result if found."
    ),
    responses={
        200: {"description": "Assessment result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Assessment not found"},
    },
)
async def get_assessment(
    assessment_id: str = Path(
        ...,
        description="Assessment identifier",
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:assess:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AssessmentResponseSchema:
    """Retrieve a stored assessment result by ID.

    Args:
        assessment_id: Unique assessment identifier.
        request: FastAPI request object.
        user: Authenticated user with assess:read permission.

    Returns:
        AssessmentResponseSchema if found.

    Raises:
        HTTPException: 404 if assessment not found.
    """
    logger.info(
        "Get assessment: user=%s assessment_id=%s",
        user.user_id,
        assessment_id,
    )

    stored = _assessment_store.get(assessment_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )

    return AssessmentResponseSchema(**stored["response"])


# ---------------------------------------------------------------------------
# POST /assess/precision
# ---------------------------------------------------------------------------


@router.post(
    "/assess/precision",
    response_model=PrecisionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Precision analysis only",
    description=(
        "Analyze the precision of a coordinate pair by counting decimal "
        "places, calculating ground resolution in metres, classifying "
        "precision level, and checking EUDR adequacy. Also detects "
        "truncation and artificial rounding patterns."
    ),
    responses={
        200: {"description": "Precision analysis result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_precision(
    body: PrecisionRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:assess:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PrecisionResponseSchema:
    """Analyze the precision of a coordinate pair.

    Counts decimal places, calculates ground resolution, classifies
    precision level, and checks EUDR adequacy (minimum 4 decimal
    places for ~11m resolution).

    Args:
        body: Request with latitude and longitude values.
        request: FastAPI request object.
        user: Authenticated user with assess:read permission.

    Returns:
        PrecisionResponseSchema with precision analysis.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Precision analysis: user=%s lat=%.10f lon=%.10f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.analyze_precision(
            latitude=body.latitude,
            longitude=body.longitude,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"precision|{body.latitude}|{body.longitude}|"
            f"{result.get('level', 'unknown')}"
        )

        logger.info(
            "Precision analysis completed: user=%s level=%s "
            "eudr_adequate=%s elapsed_ms=%.1f",
            user.user_id,
            result.get("level", "unknown"),
            result.get("eudr_adequate", False),
            elapsed * 1000,
        )

        return PrecisionResponseSchema(
            decimal_places_lat=result.get("decimal_places_lat", 0),
            decimal_places_lon=result.get("decimal_places_lon", 0),
            ground_resolution_lat_m=result.get("ground_resolution_lat_m", 0.0),
            ground_resolution_lon_m=result.get("ground_resolution_lon_m", 0.0),
            level=result.get("level", "inadequate"),
            eudr_adequate=result.get("eudr_adequate", False),
            is_truncated=result.get("is_truncated", False),
            is_rounded=result.get("is_rounded", False),
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Precision analysis failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Precision analysis failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_recommendations(
    tier_counts: Dict[str, int],
    error_counts: Dict[str, int],
    precision_counts: Dict[str, int],
) -> List[str]:
    """Generate improvement recommendations based on assessment results.

    Args:
        tier_counts: Distribution of quality tiers.
        error_counts: Error counts by type.
        precision_counts: Precision level distribution.

    Returns:
        List of recommendation strings.
    """
    recommendations: List[str] = []

    fail_count = tier_counts.get("fail", 0)
    total = sum(tier_counts.values())
    if total > 0 and fail_count > 0:
        fail_pct = fail_count / total * 100
        if fail_pct > 20:
            recommendations.append(
                f"{fail_pct:.0f}% of coordinates fail quality checks. "
                "Review data collection processes."
            )

    if error_counts.get("swapped", 0) > 0:
        recommendations.append(
            "Swapped lat/lon detected. Verify coordinate column mapping "
            "in source data."
        )

    if error_counts.get("null_island", 0) > 0:
        recommendations.append(
            "Null island coordinates (0, 0) detected. These are likely "
            "missing values, not real locations."
        )

    inadequate = precision_counts.get("inadequate", 0)
    low = precision_counts.get("low", 0)
    if (inadequate + low) > 0 and total > 0:
        low_pct = (inadequate + low) / total * 100
        if low_pct > 10:
            recommendations.append(
                f"{low_pct:.0f}% of coordinates have insufficient precision. "
                "EUDR requires minimum 4 decimal places (~11m resolution)."
            )

    if not recommendations:
        recommendations.append(
            "Coordinate quality is within acceptable ranges."
        )

    return recommendations
