# -*- coding: utf-8 -*-
"""
Classification Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for forest type classification including single-plot classification,
batch processing, stored result retrieval, and forest type reference listing.

Endpoints:
    POST /              - Classify forest type for a plot
    POST /batch         - Batch forest type classification
    GET  /{plot_id}     - Get stored classification result
    GET  /types         - List all forest types with descriptions

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_classification_engine,
    get_request_id,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    BatchClassifyRequest,
    ClassifyForestRequest,
    ForestClassificationResponse,
    ForestType,
    ForestTypeInfo,
    ForestTypesListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forest Classification"])


# ---------------------------------------------------------------------------
# Forest type reference data
# ---------------------------------------------------------------------------

_FOREST_TYPE_CATALOG: List[ForestTypeInfo] = [
    ForestTypeInfo(
        type_id="tropical_moist",
        name="Tropical Moist Forest",
        description=(
            "Broadleaf evergreen or semi-deciduous forest in tropical "
            "regions with high rainfall (>1500mm/yr). Includes lowland "
            "rainforest and montane cloud forest."
        ),
        typical_canopy_cover_pct=85.0,
        typical_height_m=35.0,
        typical_biomass_t_ha=250.0,
        climate_zones=["Af", "Am"],
        eudr_relevance="high",
    ),
    ForestTypeInfo(
        type_id="tropical_dry",
        name="Tropical Dry Forest",
        description=(
            "Deciduous or semi-deciduous forest in tropical regions with "
            "a pronounced dry season. Canopy cover varies seasonally."
        ),
        typical_canopy_cover_pct=55.0,
        typical_height_m=20.0,
        typical_biomass_t_ha=120.0,
        climate_zones=["Aw", "As"],
        eudr_relevance="high",
    ),
    ForestTypeInfo(
        type_id="subtropical",
        name="Subtropical Forest",
        description=(
            "Evergreen or mixed forest in subtropical regions with mild "
            "winters. Includes laurel forests and humid subtropical forests."
        ),
        typical_canopy_cover_pct=70.0,
        typical_height_m=25.0,
        typical_biomass_t_ha=180.0,
        climate_zones=["Cfa", "Cwa"],
        eudr_relevance="medium",
    ),
    ForestTypeInfo(
        type_id="temperate_broadleaf",
        name="Temperate Broadleaf Forest",
        description=(
            "Deciduous or mixed broadleaf forest in temperate regions. "
            "Shows strong seasonal variation in canopy cover."
        ),
        typical_canopy_cover_pct=75.0,
        typical_height_m=25.0,
        typical_biomass_t_ha=160.0,
        climate_zones=["Cfb", "Dfb"],
        eudr_relevance="medium",
    ),
    ForestTypeInfo(
        type_id="temperate_conifer",
        name="Temperate Conifer Forest",
        description=(
            "Coniferous evergreen forest in temperate regions. Includes "
            "pine, spruce, and fir-dominated stands."
        ),
        typical_canopy_cover_pct=65.0,
        typical_height_m=30.0,
        typical_biomass_t_ha=200.0,
        climate_zones=["Csb", "Dfb"],
        eudr_relevance="medium",
    ),
    ForestTypeInfo(
        type_id="boreal",
        name="Boreal Forest (Taiga)",
        description=(
            "Coniferous forest in cold northern regions dominated by "
            "spruce, larch, and pine. Low canopy density."
        ),
        typical_canopy_cover_pct=45.0,
        typical_height_m=15.0,
        typical_biomass_t_ha=80.0,
        climate_zones=["Dfc", "Dfd"],
        eudr_relevance="low",
    ),
    ForestTypeInfo(
        type_id="mangrove",
        name="Mangrove Forest",
        description=(
            "Salt-tolerant forest in coastal intertidal zones. Critical "
            "for carbon storage, coastal protection, and biodiversity."
        ),
        typical_canopy_cover_pct=70.0,
        typical_height_m=12.0,
        typical_biomass_t_ha=150.0,
        climate_zones=["Af", "Am", "Aw"],
        eudr_relevance="high",
    ),
    ForestTypeInfo(
        type_id="plantation",
        name="Plantation Forest",
        description=(
            "Planted forest managed for timber, fibre, or commodity "
            "production. Includes palm oil, rubber, and timber plantations."
        ),
        typical_canopy_cover_pct=80.0,
        typical_height_m=20.0,
        typical_biomass_t_ha=100.0,
        climate_zones=["Af", "Am", "Aw", "Cfa"],
        eudr_relevance="high",
    ),
    ForestTypeInfo(
        type_id="agroforestry",
        name="Agroforestry System",
        description=(
            "Integrated land-use system combining trees with agricultural "
            "crops or livestock. Includes shade-grown cocoa and coffee."
        ),
        typical_canopy_cover_pct=40.0,
        typical_height_m=15.0,
        typical_biomass_t_ha=60.0,
        climate_zones=["Af", "Am", "Aw"],
        eudr_relevance="high",
    ),
    ForestTypeInfo(
        type_id="secondary_growth",
        name="Secondary Growth Forest",
        description=(
            "Forest regenerating after disturbance (logging, fire, or "
            "agriculture). Characterized by younger, smaller trees."
        ),
        typical_canopy_cover_pct=55.0,
        typical_height_m=15.0,
        typical_biomass_t_ha=80.0,
        climate_zones=["Af", "Am", "Aw", "Cfa"],
        eudr_relevance="medium",
    ),
    ForestTypeInfo(
        type_id="non_forest",
        name="Non-Forest",
        description=(
            "Land not meeting FAO forest definition. Includes grassland, "
            "cropland, bare soil, water, and urban areas."
        ),
        typical_canopy_cover_pct=0.0,
        typical_height_m=0.0,
        typical_biomass_t_ha=0.0,
        climate_zones=[],
        eudr_relevance="low",
    ),
]


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ForestClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify forest type for a plot",
    description=(
        "Classify the dominant forest type for a production plot using "
        "multi-temporal satellite imagery. Supports supervised, "
        "unsupervised, object-based, deep learning, and rule-based "
        "classification methods. Returns forest type probabilities "
        "and structural characteristics."
    ),
    responses={
        200: {"description": "Forest classification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def classify_forest(
    body: ClassifyForestRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:classification:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ForestClassificationResponse:
    """Classify the dominant forest type for a plot.

    Uses the specified classification methods to determine the forest
    type from multi-temporal satellite imagery within the date range.

    Args:
        body: Classification request with plot polygon and methods.
        user: Authenticated user with classification:write permission.

    Returns:
        ForestClassificationResponse with type and probabilities.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    methods = (
        [m.value for m in body.methods]
        if body.methods
        else ["supervised", "deep_learning"]
    )

    logger.info(
        "Forest classification: user=%s plot_id=%s methods=%s "
        "dates=%s..%s",
        user.user_id,
        body.plot_id,
        methods,
        body.date_range_start,
        body.date_range_end,
    )

    try:
        engine = get_classification_engine()

        result = engine.classify(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            date_range_start=body.date_range_start,
            date_range_end=body.date_range_end,
            methods=methods,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Forest classification completed: user=%s plot_id=%s "
            "type=%s confidence=%.2f elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "primary_type", "unknown"),
            getattr(result, "confidence", 0.0),
            elapsed * 1000,
        )

        return ForestClassificationResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            primary_type=getattr(result, "primary_type", ForestType.UNKNOWN),
            secondary_type=getattr(result, "secondary_type", None),
            type_probabilities=getattr(result, "type_probabilities", {}),
            methods_used=methods,
            dominant_species_group=getattr(
                result, "dominant_species_group", None
            ),
            canopy_structure=getattr(result, "canopy_structure", "unknown"),
            is_primary_forest=getattr(result, "is_primary_forest", False),
            is_plantation=getattr(result, "is_plantation", False),
            area_ha=getattr(result, "area_ha", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            scenes_analyzed=getattr(result, "scenes_analyzed", 0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Forest classification error: user=%s plot_id=%s error=%s",
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
            "Forest classification failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forest classification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=List[ForestClassificationResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch forest type classification",
    description=(
        "Classify forest types for multiple plots in a single request. "
        "Supports up to 5,000 plots per batch."
    ),
    responses={
        200: {"description": "Batch classification results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_classify(
    body: BatchClassifyRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:classification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> List[ForestClassificationResponse]:
    """Batch forest type classification for multiple plots.

    Args:
        body: Batch request with list of classification requests.
        user: Authenticated user with classification:write permission.

    Returns:
        List of ForestClassificationResponse for each plot.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Batch classification: user=%s plots=%d",
        user.user_id,
        len(body.plots),
    )

    try:
        engine = get_classification_engine()
        results = []

        for plot_req in body.plots:
            methods = (
                [m.value for m in plot_req.methods]
                if plot_req.methods
                else ["supervised", "deep_learning"]
            )
            try:
                result = engine.classify(
                    plot_id=plot_req.plot_id,
                    polygon_wkt=plot_req.polygon_wkt,
                    date_range_start=plot_req.date_range_start,
                    date_range_end=plot_req.date_range_end,
                    methods=methods,
                )

                results.append(ForestClassificationResponse(
                    request_id=get_request_id(),
                    plot_id=plot_req.plot_id,
                    primary_type=getattr(
                        result, "primary_type", ForestType.UNKNOWN
                    ),
                    secondary_type=getattr(result, "secondary_type", None),
                    type_probabilities=getattr(
                        result, "type_probabilities", {}
                    ),
                    methods_used=methods,
                    confidence=getattr(result, "confidence", 0.0),
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(result, "provenance_hash", ""),
                ))
            except Exception as plot_exc:
                logger.warning(
                    "Batch classify: plot %s failed: %s",
                    plot_req.plot_id,
                    plot_exc,
                )
                results.append(ForestClassificationResponse(
                    request_id=get_request_id(),
                    plot_id=plot_req.plot_id,
                    primary_type=ForestType.UNKNOWN,
                    confidence=0.0,
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch classification completed: user=%s plots=%d elapsed_ms=%.1f",
            user.user_id,
            len(results),
            elapsed * 1000,
        )

        return results

    except Exception as exc:
        logger.error(
            "Batch classification failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch classification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=ForestClassificationResponse,
    summary="Get stored classification result",
    description="Retrieve the most recent stored forest classification for a plot.",
    responses={
        200: {"description": "Stored classification result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Classification not found"},
    },
)
async def get_classification(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:classification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ForestClassificationResponse:
    """Get the most recent stored classification for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with classification:read permission.

    Returns:
        ForestClassificationResponse with stored classification.

    Raises:
        HTTPException: 404 if classification not found.
    """
    logger.info(
        "Classification retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_classification_engine()
        result = engine.get_result(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No classification result found for plot {plot_id}",
            )

        return ForestClassificationResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            primary_type=getattr(result, "primary_type", ForestType.UNKNOWN),
            secondary_type=getattr(result, "secondary_type", None),
            type_probabilities=getattr(result, "type_probabilities", {}),
            methods_used=getattr(result, "methods_used", []),
            dominant_species_group=getattr(
                result, "dominant_species_group", None
            ),
            canopy_structure=getattr(result, "canopy_structure", "unknown"),
            is_primary_forest=getattr(result, "is_primary_forest", False),
            is_plantation=getattr(result, "is_plantation", False),
            area_ha=getattr(result, "area_ha", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            scenes_analyzed=getattr(result, "scenes_analyzed", 0),
            timestamp=getattr(result, "timestamp", datetime.now(timezone.utc)),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Classification retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Classification retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /types
# ---------------------------------------------------------------------------


@router.get(
    "/types",
    response_model=ForestTypesListResponse,
    summary="List all forest types",
    description=(
        "List all supported forest types with descriptions, typical "
        "canopy cover, height, biomass, climate zones, and EUDR relevance."
    ),
    responses={
        200: {"description": "Forest types catalog"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_forest_types(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:classification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ForestTypesListResponse:
    """List all supported forest types with descriptions.

    Args:
        user: Authenticated user with classification:read permission.

    Returns:
        ForestTypesListResponse with all forest type definitions.
    """
    logger.info(
        "Forest types listing: user=%s",
        user.user_id,
    )

    return ForestTypesListResponse(
        request_id=get_request_id(),
        types=_FOREST_TYPE_CATALOG,
        total=len(_FOREST_TYPE_CATALOG),
    )
