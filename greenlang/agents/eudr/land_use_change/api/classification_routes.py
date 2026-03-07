# -*- coding: utf-8 -*-
"""
Classification Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for multi-class land use classification including single-plot
classification, batch processing, stored result retrieval, classification
history, and two-date comparison.

Endpoints:
    POST /              - Classify land use for a single plot
    POST /batch         - Batch land use classification
    GET  /{plot_id}     - Get stored classification result
    GET  /{plot_id}/history - Get classification history
    POST /compare       - Compare classification between two dates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.land_use_change.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_classifier_engine,
    get_land_use_service,
    get_pagination,
    get_request_id,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    ClassificationBatchResponse,
    ClassificationCompareResponse,
    ClassificationHistoryEntry,
    ClassificationHistoryResponse,
    ClassificationResult,
    ClassifyBatchRequest,
    ClassifyCompareRequest,
    ClassifyRequest,
    LandUseCategory,
    PaginatedMeta,
    TransitionType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Land Use Classification"])

# ---------------------------------------------------------------------------
# In-memory result store (replaced by database in production)
# ---------------------------------------------------------------------------

_classification_store: Dict[str, Dict[str, Any]] = {}


def _get_classification_store() -> Dict[str, Dict[str, Any]]:
    """Return the classification store. Replaceable for testing."""
    return _classification_store


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Classify land use for a single plot",
    description=(
        "Classify the land use category for a single geographic location "
        "or plot polygon using the specified method. Returns one of 10 "
        "IPCC-aligned land use categories (forest, shrubland, grassland, "
        "cropland, wetland, water, urban, bare_soil, snow_ice, other) "
        "with confidence scores and spectral indices. Supports spectral, "
        "vegetation_index, phenology, texture, and ensemble methods."
    ),
    responses={
        200: {"description": "Classification result with confidence"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def classify_land_use(
    body: ClassifyRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:classification:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ClassificationResult:
    """Classify land use for a single plot.

    Invokes the LandUseClassifier engine with the specified method
    and optional commodity context for targeted classification.

    Args:
        body: Classification request with coordinates, date, and method.
        user: Authenticated user with classification:write permission.

    Returns:
        ClassificationResult with category, confidence, and spectral indices.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Land use classification: user=%s plot=%s lat=%.6f lon=%.6f "
        "date=%s method=%s commodity=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.date,
        body.method.value,
        body.commodity_context.value if body.commodity_context else "none",
    )

    try:
        engine = get_classifier_engine()

        result = engine.classify(
            latitude=body.latitude,
            longitude=body.longitude,
            target_date=body.date,
            method=body.method.value,
            commodity_context=(
                body.commodity_context.value
                if body.commodity_context
                else None
            ),
            polygon_wkt=body.polygon_wkt,
        )

        elapsed = time.monotonic() - start

        response = ClassificationResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            land_use_category=getattr(
                result, "category", LandUseCategory.OTHER
            ),
            sub_category=getattr(result, "sub_category", None),
            confidence=getattr(result, "confidence", 0.0),
            method=body.method,
            class_probabilities=getattr(
                result, "class_probabilities", {}
            ),
            spectral_indices=getattr(result, "spectral_indices", None),
            latitude=body.latitude,
            longitude=body.longitude,
            area_ha=getattr(result, "area_ha", None),
            imagery_date=getattr(result, "imagery_date", None),
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_classification_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "response_data": response.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Classification completed: user=%s plot=%s category=%s "
            "confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            response.land_use_category.value,
            response.confidence,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Classification error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Classification failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification processing failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=ClassificationBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch land use classification",
    description=(
        "Classify land use for multiple plots in a single request. "
        "Supports up to 5000 plots per batch. Results include per-plot "
        "classification with confidence scores and aggregate statistics."
    ),
    responses={
        200: {"description": "Batch classification results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def classify_batch(
    body: ClassifyBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:classification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ClassificationBatchResponse:
    """Batch classify land use for multiple plots.

    Processes each plot sequentially through the classifier engine.
    Individual plot failures do not block the entire batch.

    Args:
        body: Batch request with list of plots and optional global settings.
        user: Authenticated user with classification:write permission.

    Returns:
        ClassificationBatchResponse with results and aggregate statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch classification: user=%s plots=%d method=%s",
        user.user_id,
        total,
        body.method.value,
    )

    results: List[ClassificationResult] = []
    successful = 0
    failed = 0
    category_counts: Dict[str, int] = {}
    confidence_sum = 0.0

    try:
        engine = get_classifier_engine()
        store = _get_classification_store()

        for plot_req in body.plots:
            plot_id = plot_req.plot_id or f"luc-{uuid.uuid4().hex[:12]}"
            target_date = body.date or plot_req.date

            try:
                result = engine.classify(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    target_date=target_date,
                    method=body.method.value,
                    commodity_context=(
                        plot_req.commodity_context.value
                        if plot_req.commodity_context
                        else None
                    ),
                    polygon_wkt=plot_req.polygon_wkt,
                )

                category = getattr(
                    result, "category", LandUseCategory.OTHER
                )
                confidence = getattr(result, "confidence", 0.0)

                classification = ClassificationResult(
                    request_id=get_request_id(),
                    plot_id=plot_id,
                    land_use_category=category,
                    sub_category=getattr(result, "sub_category", None),
                    confidence=confidence,
                    method=body.method,
                    class_probabilities=getattr(
                        result, "class_probabilities", {}
                    ),
                    spectral_indices=getattr(
                        result, "spectral_indices", None
                    ),
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    area_ha=getattr(result, "area_ha", None),
                    imagery_date=getattr(result, "imagery_date", None),
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(classification)
                successful += 1
                confidence_sum += confidence

                cat_key = (
                    category.value
                    if hasattr(category, "value")
                    else str(category)
                )
                category_counts[cat_key] = (
                    category_counts.get(cat_key, 0) + 1
                )

                store[plot_id] = {
                    "plot_id": plot_id,
                    "response_data": classification.model_dump(mode="json"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch classification failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start
        mean_confidence = (
            confidence_sum / successful if successful > 0 else 0.0
        )

        logger.info(
            "Batch classification completed: user=%s total=%d "
            "successful=%d failed=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            elapsed * 1000,
        )

        return ClassificationBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            category_distribution=category_counts,
            mean_confidence=mean_confidence,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch classification failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=ClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored classification result",
    description=(
        "Retrieve a previously computed classification result by plot ID."
    ),
    responses={
        200: {"description": "Classification result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_classification(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:classification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ClassificationResult:
    """Retrieve a stored classification result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with classification:read permission.

    Returns:
        ClassificationResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_classification_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No classification result found for plot_id '{plot_id}'",
        )

    record = store[plot_id]
    return ClassificationResult(**record["response_data"])


# ---------------------------------------------------------------------------
# GET /{plot_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/history",
    response_model=ClassificationHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get classification history",
    description=(
        "Retrieve classification history over time for a plot, showing "
        "how land use has changed across multiple observation dates."
    ),
    responses={
        200: {"description": "Classification history with time series"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_classification_history(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:classification:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
) -> ClassificationHistoryResponse:
    """Retrieve classification history for a plot.

    Queries the analysis engines for historical classification data
    at the specified plot location. Returns a time series of
    classifications with trend analysis.

    Args:
        plot_id: Plot identifier to look up history for.
        user: Authenticated user with classification:read permission.
        pagination: Pagination parameters.

    Returns:
        ClassificationHistoryResponse with time series entries.

    Raises:
        HTTPException: 404 if plot_id not found, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = validate_plot_id(plot_id)

    logger.info(
        "Classification history: user=%s plot=%s limit=%d offset=%d",
        user.user_id,
        plot_id,
        pagination.limit,
        pagination.offset,
    )

    try:
        service = get_land_use_service()

        result = service.get_classification_history(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        entries_raw = getattr(result, "entries", [])
        entries = []
        for entry in entries_raw:
            entries.append(
                ClassificationHistoryEntry(
                    date=getattr(entry, "date", None),
                    land_use_category=getattr(
                        entry, "category", LandUseCategory.OTHER
                    ),
                    sub_category=getattr(entry, "sub_category", None),
                    confidence=getattr(entry, "confidence", 0.0),
                    method=getattr(entry, "method", "ensemble"),
                    source=getattr(entry, "source", ""),
                )
            )

        total_obs = getattr(result, "total_observations", len(entries))
        elapsed = time.monotonic() - start

        logger.info(
            "Classification history completed: user=%s plot=%s "
            "entries=%d elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            len(entries),
            elapsed * 1000,
        )

        return ClassificationHistoryResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            entries=entries,
            dominant_category=getattr(
                result, "dominant_category", None
            ),
            transitions_count=getattr(
                result, "transitions_count", 0
            ),
            total_observations=total_obs,
            meta=PaginatedMeta(
                total=total_obs,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(
                    pagination.offset + pagination.limit < total_obs
                ),
            ),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Classification history failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification history retrieval failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=ClassificationCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare classification between two dates",
    description=(
        "Classify land use at a single location for two different dates "
        "and compare the results. Detects transitions and reports the "
        "transition type if land use has changed."
    ),
    responses={
        200: {"description": "Comparison result with transition detection"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_classification(
    body: ClassifyCompareRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:classification:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ClassificationCompareResponse:
    """Compare land use classification between two dates.

    Runs the classifier for both dates and determines whether a
    land use transition has occurred.

    Args:
        body: Comparison request with coordinates and two dates.
        user: Authenticated user with classification:write permission.

    Returns:
        ClassificationCompareResponse with both results and transition info.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-cmp-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Classification comparison: user=%s plot=%s lat=%.6f lon=%.6f "
        "date1=%s date2=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.date1,
        body.date2,
    )

    try:
        engine = get_classifier_engine()

        # Classify at date1
        result1 = engine.classify(
            latitude=body.latitude,
            longitude=body.longitude,
            target_date=body.date1,
            method=body.method.value,
            polygon_wkt=body.polygon_wkt,
        )

        # Classify at date2
        result2 = engine.classify(
            latitude=body.latitude,
            longitude=body.longitude,
            target_date=body.date2,
            method=body.method.value,
            polygon_wkt=body.polygon_wkt,
        )

        cat1 = getattr(result1, "category", LandUseCategory.OTHER)
        cat2 = getattr(result2, "category", LandUseCategory.OTHER)
        conf1 = getattr(result1, "confidence", 0.0)
        conf2 = getattr(result2, "confidence", 0.0)

        # Determine if transition occurred
        cat1_val = cat1.value if hasattr(cat1, "value") else str(cat1)
        cat2_val = cat2.value if hasattr(cat2, "value") else str(cat2)
        transition_detected = cat1_val != cat2_val

        # Determine transition type
        transition_type = None
        if transition_detected:
            if cat1_val == "forest" and cat2_val != "forest":
                transition_type = TransitionType.DEFORESTATION
            elif cat1_val != "forest" and cat2_val == "forest":
                transition_type = TransitionType.REFORESTATION
            elif cat2_val == "urban":
                transition_type = TransitionType.URBANIZATION
            elif cat2_val == "cropland":
                transition_type = TransitionType.AGRICULTURAL_EXPANSION
            else:
                transition_type = TransitionType.UNKNOWN

        date1_result = ClassificationResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            land_use_category=cat1,
            sub_category=getattr(result1, "sub_category", None),
            confidence=conf1,
            method=body.method,
            class_probabilities=getattr(
                result1, "class_probabilities", {}
            ),
            spectral_indices=getattr(result1, "spectral_indices", None),
            latitude=body.latitude,
            longitude=body.longitude,
            imagery_date=getattr(result1, "imagery_date", None),
            data_sources=getattr(result1, "data_sources", []),
            provenance_hash=getattr(result1, "provenance_hash", ""),
        )

        date2_result = ClassificationResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            land_use_category=cat2,
            sub_category=getattr(result2, "sub_category", None),
            confidence=conf2,
            method=body.method,
            class_probabilities=getattr(
                result2, "class_probabilities", {}
            ),
            spectral_indices=getattr(result2, "spectral_indices", None),
            latitude=body.latitude,
            longitude=body.longitude,
            imagery_date=getattr(result2, "imagery_date", None),
            data_sources=getattr(result2, "data_sources", []),
            provenance_hash=getattr(result2, "provenance_hash", ""),
        )

        elapsed = time.monotonic() - start

        logger.info(
            "Classification comparison completed: user=%s plot=%s "
            "cat1=%s cat2=%s transition=%s type=%s elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            cat1_val,
            cat2_val,
            transition_detected,
            transition_type.value if transition_type else "none",
            elapsed * 1000,
        )

        return ClassificationCompareResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            date1_result=date1_result,
            date2_result=date2_result,
            transition_detected=transition_detected,
            transition_type=transition_type,
            confidence=min(conf1, conf2),
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Comparison error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Comparison failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification comparison failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
