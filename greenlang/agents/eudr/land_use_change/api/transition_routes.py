# -*- coding: utf-8 -*-
"""
Transition Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for land use transition detection including single-plot detection,
batch processing, stored result retrieval, transition matrix generation,
and transition type reference listing.

Endpoints:
    POST /detect        - Detect transition for a single plot
    POST /batch         - Batch transition detection
    GET  /{plot_id}     - Get stored transition result
    POST /matrix        - Generate transition matrix for a region
    GET  /types         - List all transition types with descriptions

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
    get_land_use_service,
    get_request_id,
    get_transition_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    LandUseCategory,
    TopTransition,
    TransitionBatchRequest,
    TransitionBatchResponse,
    TransitionDetectRequest,
    TransitionEvidence,
    TransitionMatrixCell,
    TransitionMatrixRequest,
    TransitionMatrixResponse,
    TransitionResult,
    TransitionType,
    TransitionTypeInfo,
    TransitionTypesListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Transition Detection"])

# ---------------------------------------------------------------------------
# In-memory result store (replaced by database in production)
# ---------------------------------------------------------------------------

_transition_store: Dict[str, Dict[str, Any]] = {}


def _get_transition_store() -> Dict[str, Dict[str, Any]]:
    """Return the transition store. Replaceable for testing."""
    return _transition_store


# ---------------------------------------------------------------------------
# Transition type catalog
# ---------------------------------------------------------------------------

_TRANSITION_TYPE_CATALOG: List[TransitionTypeInfo] = [
    TransitionTypeInfo(
        type_id="deforestation",
        name="Deforestation",
        description=(
            "Conversion of forest land to non-forest land use. "
            "Includes clear-cutting, slash-and-burn, and gradual "
            "forest removal for agriculture or development."
        ),
        from_categories=["forest"],
        to_categories=[
            "cropland", "grassland", "urban", "bare_soil", "other",
        ],
        eudr_relevance="high",
        severity="critical",
    ),
    TransitionTypeInfo(
        type_id="degradation",
        name="Forest Degradation",
        description=(
            "Reduction in forest quality, density, or carbon stock "
            "without complete conversion to non-forest. Includes "
            "selective logging, fire damage, and canopy thinning."
        ),
        from_categories=["forest"],
        to_categories=["forest"],
        eudr_relevance="high",
        severity="high",
    ),
    TransitionTypeInfo(
        type_id="afforestation",
        name="Afforestation",
        description=(
            "Establishment of forest on land that has not been "
            "forested for a significant period (typically 50+ years)."
        ),
        from_categories=[
            "grassland", "cropland", "bare_soil", "shrubland",
        ],
        to_categories=["forest"],
        eudr_relevance="low",
        severity="low",
    ),
    TransitionTypeInfo(
        type_id="reforestation",
        name="Reforestation",
        description=(
            "Re-establishment of forest on land that was previously "
            "forested. Includes natural regeneration and planting."
        ),
        from_categories=[
            "grassland", "cropland", "shrubland", "bare_soil",
        ],
        to_categories=["forest"],
        eudr_relevance="medium",
        severity="low",
    ),
    TransitionTypeInfo(
        type_id="agricultural_expansion",
        name="Agricultural Expansion",
        description=(
            "Conversion of natural or semi-natural land to "
            "agricultural use, including both cropland and pasture."
        ),
        from_categories=[
            "forest", "shrubland", "grassland", "wetland",
        ],
        to_categories=["cropland"],
        eudr_relevance="high",
        severity="high",
    ),
    TransitionTypeInfo(
        type_id="urbanization",
        name="Urbanization",
        description=(
            "Conversion of any land use to urban or built-up area. "
            "Includes residential, commercial, industrial, and "
            "infrastructure development."
        ),
        from_categories=[
            "forest", "cropland", "grassland", "shrubland",
            "wetland", "bare_soil",
        ],
        to_categories=["urban"],
        eudr_relevance="medium",
        severity="high",
    ),
    TransitionTypeInfo(
        type_id="wetland_conversion",
        name="Wetland Conversion",
        description=(
            "Drainage or filling of wetland areas for agriculture, "
            "development, or other land uses."
        ),
        from_categories=["wetland"],
        to_categories=[
            "cropland", "grassland", "urban", "bare_soil",
        ],
        eudr_relevance="medium",
        severity="high",
    ),
    TransitionTypeInfo(
        type_id="cropland_abandonment",
        name="Cropland Abandonment",
        description=(
            "Abandonment of agricultural land, typically followed "
            "by natural succession to grassland or shrubland."
        ),
        from_categories=["cropland"],
        to_categories=["grassland", "shrubland", "forest"],
        eudr_relevance="low",
        severity="low",
    ),
    TransitionTypeInfo(
        type_id="intensification",
        name="Agricultural Intensification",
        description=(
            "Change in agricultural management intensity without "
            "change in land use category. Includes conversion from "
            "extensive to intensive farming."
        ),
        from_categories=["cropland", "grassland"],
        to_categories=["cropland", "grassland"],
        eudr_relevance="medium",
        severity="medium",
    ),
    TransitionTypeInfo(
        type_id="stable",
        name="Stable (No Change)",
        description=(
            "No land use transition detected between the two "
            "observation dates. Land use category remains unchanged."
        ),
        from_categories=[],
        to_categories=[],
        eudr_relevance="low",
        severity="low",
    ),
    TransitionTypeInfo(
        type_id="unknown",
        name="Unknown Transition",
        description=(
            "Transition detected but type could not be determined "
            "with sufficient confidence. Requires manual review."
        ),
        from_categories=[],
        to_categories=[],
        eudr_relevance="medium",
        severity="medium",
    ),
]


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=TransitionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect land use transition",
    description=(
        "Detect whether a land use transition has occurred at a plot "
        "between two dates. Returns the from/to land use categories, "
        "transition type, estimated transition date, and supporting "
        "evidence. Relevant for EUDR compliance when checking post-"
        "cutoff deforestation or degradation."
    ),
    responses={
        200: {"description": "Transition detection result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_transition(
    body: TransitionDetectRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:transitions:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TransitionResult:
    """Detect land use transition for a single plot.

    Args:
        body: Transition detection request with coordinates and date range.
        user: Authenticated user with transitions:write permission.

    Returns:
        TransitionResult with from/to classes and evidence.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-tr-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Transition detection: user=%s plot=%s lat=%.6f lon=%.6f "
        "from=%s to=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.date_from,
        body.date_to,
    )

    try:
        engine = get_transition_engine()

        result = engine.detect(
            latitude=body.latitude,
            longitude=body.longitude,
            date_from=body.date_from,
            date_to=body.date_to,
            polygon_wkt=body.polygon_wkt,
            min_area_ha=body.min_area_ha,
        )

        elapsed = time.monotonic() - start

        # Build evidence items
        evidence = []
        raw_evidence = getattr(result, "evidence", [])
        for item in raw_evidence:
            evidence.append(
                TransitionEvidence(
                    evidence_type=getattr(item, "evidence_type", ""),
                    description=getattr(item, "description", ""),
                    date=getattr(item, "date", None),
                    source=getattr(item, "source", ""),
                    value=getattr(item, "value", None),
                    unit=getattr(item, "unit", None),
                    confidence=getattr(item, "confidence", 0.0),
                )
            )

        from_class = getattr(
            result, "from_class", LandUseCategory.OTHER
        )
        to_class = getattr(
            result, "to_class", LandUseCategory.OTHER
        )
        transition_type = getattr(
            result, "transition_type", TransitionType.STABLE
        )

        response = TransitionResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            from_class=from_class,
            to_class=to_class,
            transition_type=transition_type,
            date_range={
                "from": str(body.date_from),
                "to": str(body.date_to),
            },
            estimated_transition_date=getattr(
                result, "estimated_transition_date", None
            ),
            transition_area_ha=getattr(
                result, "transition_area_ha", 0.0
            ),
            confidence=getattr(result, "confidence", 0.0),
            evidence=evidence,
            is_eudr_relevant=getattr(
                result, "is_eudr_relevant", False
            ),
            latitude=body.latitude,
            longitude=body.longitude,
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_transition_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "response_data": response.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Transition detection completed: user=%s plot=%s "
            "from=%s to=%s type=%s confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            getattr(from_class, "value", from_class),
            getattr(to_class, "value", to_class),
            getattr(transition_type, "value", transition_type),
            response.confidence,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Transition detection error: user=%s plot=%s error=%s",
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
            "Transition detection failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transition detection failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=TransitionBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch transition detection",
    description=(
        "Detect land use transitions for multiple plots in a single "
        "request. Supports up to 5000 plots per batch. Returns "
        "per-plot results and aggregate statistics including "
        "deforestation and degradation counts."
    ),
    responses={
        200: {"description": "Batch transition results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_batch(
    body: TransitionBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:transitions:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> TransitionBatchResponse:
    """Batch detect transitions for multiple plots.

    Args:
        body: Batch request with list of plots and optional global dates.
        user: Authenticated user with transitions:write permission.

    Returns:
        TransitionBatchResponse with results and aggregate statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch transition detection: user=%s plots=%d",
        user.user_id,
        total,
    )

    results: List[TransitionResult] = []
    successful = 0
    failed = 0
    deforestation_count = 0
    degradation_count = 0
    stable_count = 0

    try:
        engine = get_transition_engine()
        store = _get_transition_store()

        for plot_req in body.plots:
            plot_id = plot_req.plot_id or f"luc-tr-{uuid.uuid4().hex[:12]}"
            date_from = body.date_from or plot_req.date_from
            date_to = body.date_to or plot_req.date_to

            try:
                result = engine.detect(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    date_from=date_from,
                    date_to=date_to,
                    polygon_wkt=plot_req.polygon_wkt,
                    min_area_ha=plot_req.min_area_ha,
                )

                transition_type = getattr(
                    result, "transition_type", TransitionType.STABLE
                )
                tt_val = (
                    transition_type.value
                    if hasattr(transition_type, "value")
                    else str(transition_type)
                )

                transition = TransitionResult(
                    request_id=get_request_id(),
                    plot_id=plot_id,
                    from_class=getattr(
                        result, "from_class", LandUseCategory.OTHER
                    ),
                    to_class=getattr(
                        result, "to_class", LandUseCategory.OTHER
                    ),
                    transition_type=transition_type,
                    date_range={
                        "from": str(date_from),
                        "to": str(date_to),
                    },
                    estimated_transition_date=getattr(
                        result, "estimated_transition_date", None
                    ),
                    transition_area_ha=getattr(
                        result, "transition_area_ha", 0.0
                    ),
                    confidence=getattr(result, "confidence", 0.0),
                    is_eudr_relevant=getattr(
                        result, "is_eudr_relevant", False
                    ),
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(transition)
                successful += 1

                if tt_val == "deforestation":
                    deforestation_count += 1
                elif tt_val == "degradation":
                    degradation_count += 1
                elif tt_val == "stable":
                    stable_count += 1

                store[plot_id] = {
                    "plot_id": plot_id,
                    "response_data": transition.model_dump(mode="json"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch transition failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start

        logger.info(
            "Batch transition completed: user=%s total=%d "
            "successful=%d failed=%d deforestation=%d degradation=%d "
            "stable=%d elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            deforestation_count,
            degradation_count,
            stable_count,
            elapsed * 1000,
        )

        return TransitionBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            deforestation_count=deforestation_count,
            degradation_count=degradation_count,
            stable_count=stable_count,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch transition failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch transition detection failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=TransitionResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored transition result",
    description="Retrieve a previously computed transition result by plot ID.",
    responses={
        200: {"description": "Transition result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_transition(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:transitions:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TransitionResult:
    """Retrieve a stored transition result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with transitions:read permission.

    Returns:
        TransitionResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_transition_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No transition result found for plot_id '{plot_id}'"
            ),
        )

    record = store[plot_id]
    return TransitionResult(**record["response_data"])


# ---------------------------------------------------------------------------
# POST /matrix
# ---------------------------------------------------------------------------


@router.post(
    "/matrix",
    response_model=TransitionMatrixResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate transition matrix",
    description=(
        "Generate a land use transition matrix for a geographic region "
        "defined by a bounding box. Shows area-based transition counts "
        "between all land use categories for the specified date range."
    ),
    responses={
        200: {"description": "Transition matrix with area statistics"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_transition_matrix(
    body: TransitionMatrixRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:transitions:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> TransitionMatrixResponse:
    """Generate a transition matrix for a region.

    Args:
        body: Matrix request with bounding box and date range.
        user: Authenticated user with transitions:write permission.

    Returns:
        TransitionMatrixResponse with matrix cells and statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Transition matrix: user=%s bounds=%s from=%s to=%s res=%dm",
        user.user_id,
        body.region_bounds,
        body.date_from,
        body.date_to,
        body.resolution_m,
    )

    try:
        service = get_land_use_service()

        result = service.generate_transition_matrix(
            region_bounds=body.region_bounds,
            date_from=body.date_from,
            date_to=body.date_to,
            resolution_m=body.resolution_m,
            categories=(
                [c.value for c in body.categories]
                if body.categories
                else None
            ),
        )

        # Build matrix cells
        matrix_cells = []
        raw_matrix = getattr(result, "matrix", [])
        for cell in raw_matrix:
            matrix_cells.append(
                TransitionMatrixCell(
                    from_class=getattr(cell, "from_class", ""),
                    to_class=getattr(cell, "to_class", ""),
                    area_ha=getattr(cell, "area_ha", 0.0),
                    pixel_count=getattr(cell, "pixel_count", 0),
                    percentage=getattr(cell, "percentage", 0.0),
                )
            )

        # Build top transitions
        top_transitions = []
        raw_top = getattr(result, "top_transitions", [])
        for tr in raw_top:
            top_transitions.append(
                TopTransition(
                    from_class=getattr(tr, "from_class", ""),
                    to_class=getattr(tr, "to_class", ""),
                    area_ha=getattr(tr, "area_ha", 0.0),
                    percentage=getattr(tr, "percentage", 0.0),
                )
            )

        elapsed = time.monotonic() - start

        logger.info(
            "Transition matrix completed: user=%s cells=%d "
            "total_area=%.1fha transitions=%d elapsed_ms=%.1f",
            user.user_id,
            len(matrix_cells),
            getattr(result, "total_area_ha", 0.0),
            getattr(result, "transitions_detected", 0),
            elapsed * 1000,
        )

        return TransitionMatrixResponse(
            request_id=get_request_id(),
            matrix=matrix_cells,
            total_area_ha=getattr(result, "total_area_ha", 0.0),
            transitions_detected=getattr(
                result, "transitions_detected", 0
            ),
            top_transitions=top_transitions,
            date_from=body.date_from,
            date_to=body.date_to,
            resolution_m=body.resolution_m,
            region_bounds=body.region_bounds,
            deforestation_area_ha=getattr(
                result, "deforestation_area_ha", 0.0
            ),
            net_forest_change_ha=getattr(
                result, "net_forest_change_ha", 0.0
            ),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Transition matrix error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Transition matrix failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transition matrix generation failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /types
# ---------------------------------------------------------------------------


@router.get(
    "/types",
    response_model=TransitionTypesListResponse,
    status_code=status.HTTP_200_OK,
    summary="List transition types",
    description=(
        "List all supported land use transition types with descriptions, "
        "from/to categories, EUDR relevance, and severity levels."
    ),
    responses={
        200: {"description": "List of transition types"},
    },
)
async def list_transition_types(
    request: Request,
) -> TransitionTypesListResponse:
    """Return the catalog of supported transition types.

    No authentication required for reference data.

    Returns:
        TransitionTypesListResponse with all transition type definitions.
    """
    return TransitionTypesListResponse(
        request_id=get_request_id(),
        types=_TRANSITION_TYPE_CATALOG,
        total=len(_TRANSITION_TYPE_CATALOG),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
