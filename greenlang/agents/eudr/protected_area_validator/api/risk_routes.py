# -*- coding: utf-8 -*-
"""
Risk Scoring Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for calculating protected area risk scores, generating risk
heatmaps, providing risk summaries, and monitoring high-risk proximity alerts.

Endpoints:
    POST /risk/score            - Calculate risk score for a plot
    GET  /risk/heatmap          - Get risk heatmap data
    GET  /risk/summary          - Get risk summary statistics
    GET  /risk/proximity-alerts - Get high-risk proximity alerts

Auth: eudr-pav:risk:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, RiskScorer Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_risk_scorer,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    ErrorResponse,
    GeoBoundingBoxSchema,
    MetadataSchema,
    PaginatedMeta,
    ProximityAlertEntry,
    ProximityAlertsResponse,
    ProvenanceInfo,
    RiskHeatmapCell,
    RiskHeatmapResponse,
    RiskLevelEnum,
    RiskScoreBreakdown,
    RiskScoreRequest,
    RiskScoreResponse,
    RiskSummaryByCategory,
    RiskSummaryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk Scoring"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /risk/score
# ---------------------------------------------------------------------------


@router.post(
    "/score",
    response_model=RiskScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate protected area risk score for a plot",
    description=(
        "Calculate a composite risk score for a supply chain plot based on "
        "proximity to protected areas, overlap extent, designation strength, "
        "IUCN category restrictiveness, and biodiversity value. Scores range "
        "from 0 (negligible) to 1 (critical)."
    ),
    responses={
        200: {"description": "Risk score calculated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def calculate_risk_score(
    request: Request,
    body: RiskScoreRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:risk:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> RiskScoreResponse:
    """Calculate protected area risk score for a plot.

    Args:
        body: Risk score request with plot coordinates and parameters.
        user: Authenticated user with risk:create permission.

    Returns:
        RiskScoreResponse with score and breakdown.
    """
    start = time.monotonic()

    try:
        engine = get_risk_scorer()
        result = engine.calculate_score(
            plot_id=body.plot_id,
            latitude=float(body.plot_center.latitude),
            longitude=float(body.plot_center.longitude),
            plot_boundary=(
                [{"latitude": float(p.latitude), "longitude": float(p.longitude)}
                 for p in body.plot_boundary.coordinates]
                if body.plot_boundary else None
            ),
            plot_area_ha=float(body.plot_area_ha) if body.plot_area_ha else None,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            include_breakdown=body.include_breakdown,
            custom_weights=(
                {k: float(v) for k, v in body.custom_weights.items()}
                if body.custom_weights else None
            ),
        )

        breakdown = None
        if body.include_breakdown and result.get("breakdown"):
            b = result["breakdown"]
            breakdown = RiskScoreBreakdown(
                proximity_score=Decimal(str(b.get("proximity_score", 0))),
                proximity_weight=Decimal(str(b.get("proximity_weight", 0.3))),
                overlap_score=Decimal(str(b.get("overlap_score", 0))),
                overlap_weight=Decimal(str(b.get("overlap_weight", 0.25))),
                designation_score=Decimal(str(b.get("designation_score", 0))),
                designation_weight=Decimal(str(b.get("designation_weight", 0.15))),
                iucn_category_score=Decimal(str(b.get("iucn_category_score", 0))),
                iucn_weight=Decimal(str(b.get("iucn_weight", 0.15))),
                biodiversity_score=Decimal(str(b.get("biodiversity_score", 0))),
                biodiversity_weight=Decimal(str(b.get("biodiversity_weight", 0.15))),
                weighted_total=Decimal(str(b.get("weighted_total", 0))),
                multiplier_applied=Decimal(str(b.get("multiplier_applied", 1)))
                if b.get("multiplier_applied") is not None else None,
                final_score=Decimal(str(b.get("final_score", 0))),
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"risk_score:{body.plot_id}", str(result.get("risk_score", 0)),
        )

        logger.info(
            "Risk scored: plot_id=%s score=%s level=%s operator=%s",
            body.plot_id,
            result.get("risk_score", 0),
            result.get("risk_level", "negligible"),
            user.operator_id or user.user_id,
        )

        return RiskScoreResponse(
            plot_id=body.plot_id,
            risk_level=RiskLevelEnum(result.get("risk_level", "negligible")),
            risk_score=Decimal(str(result.get("risk_score", 0))),
            breakdown=breakdown,
            nearest_area_name=result.get("nearest_area_name"),
            nearest_area_distance_km=Decimal(str(result.get("nearest_area_distance_km", 0)))
            if result.get("nearest_area_distance_km") is not None else None,
            total_areas_in_range=result.get("total_areas_in_range", 0),
            risk_factors=result.get("risk_factors", []),
            classification_reason=result.get("classification_reason", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RiskScorer", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Risk score calculation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk score calculation failed",
        )


# ---------------------------------------------------------------------------
# GET /risk/heatmap
# ---------------------------------------------------------------------------


@router.get(
    "/heatmap",
    response_model=RiskHeatmapResponse,
    summary="Get risk heatmap data",
    description=(
        "Generate a grid-based risk heatmap for a geographic bounding box. "
        "Each grid cell contains a risk score based on proximity to and "
        "density of protected areas."
    ),
    responses={
        200: {"description": "Heatmap data generated"},
        400: {"model": ErrorResponse, "description": "Invalid bounding box"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def get_risk_heatmap(
    request: Request,
    min_lat: Decimal = Query(..., ge=Decimal("-90"), le=Decimal("90"), description="Southern boundary"),
    max_lat: Decimal = Query(..., ge=Decimal("-90"), le=Decimal("90"), description="Northern boundary"),
    min_lon: Decimal = Query(..., ge=Decimal("-180"), le=Decimal("180"), description="Western boundary"),
    max_lon: Decimal = Query(..., ge=Decimal("-180"), le=Decimal("180"), description="Eastern boundary"),
    resolution_km: Decimal = Query(
        default=Decimal("10"), gt=Decimal("0"), le=Decimal("100"),
        description="Grid cell resolution in km",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-pav:risk:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> RiskHeatmapResponse:
    """Generate risk heatmap data for a bounding box.

    Args:
        min_lat: Southern boundary latitude.
        max_lat: Northern boundary latitude.
        min_lon: Western boundary longitude.
        max_lon: Eastern boundary longitude.
        resolution_km: Grid cell size in km.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskHeatmapResponse with grid cell data.
    """
    start = time.monotonic()

    if max_lat < min_lat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_lat must be >= min_lat",
        )

    try:
        engine = get_risk_scorer()
        result = engine.generate_heatmap(
            min_lat=float(min_lat),
            max_lat=float(max_lat),
            min_lon=float(min_lon),
            max_lon=float(max_lon),
            resolution_km=float(resolution_km),
        )

        cells = []
        high_risk_count = 0
        for c in result.get("cells", []):
            risk_level = RiskLevelEnum(c.get("risk_level", "negligible"))
            cells.append(
                RiskHeatmapCell(
                    latitude=Decimal(str(c.get("latitude", 0))),
                    longitude=Decimal(str(c.get("longitude", 0))),
                    risk_score=Decimal(str(c.get("risk_score", 0))),
                    risk_level=risk_level,
                    nearest_area_km=Decimal(str(c.get("nearest_area_km", 0)))
                    if c.get("nearest_area_km") is not None else None,
                    protected_areas_count=c.get("protected_areas_count", 0),
                )
            )
            if risk_level in (RiskLevelEnum.HIGH, RiskLevelEnum.CRITICAL):
                high_risk_count += 1

        bbox = GeoBoundingBoxSchema(
            min_latitude=min_lat,
            max_latitude=max_lat,
            min_longitude=min_lon,
            max_longitude=max_lon,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"heatmap:{min_lat},{min_lon}:{max_lat},{max_lon}",
            str(len(cells)),
        )

        logger.info(
            "Risk heatmap: cells=%d high_risk=%d resolution=%skm operator=%s",
            len(cells),
            high_risk_count,
            resolution_km,
            user.operator_id or user.user_id,
        )

        return RiskHeatmapResponse(
            cells=cells,
            total_cells=len(cells),
            grid_resolution_km=resolution_km,
            bounding_box=bbox,
            high_risk_cells=high_risk_count,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RiskScorer", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Risk heatmap generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk heatmap generation failed",
        )


# ---------------------------------------------------------------------------
# GET /risk/summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=RiskSummaryResponse,
    summary="Get risk summary statistics",
    description=(
        "Retrieve aggregate risk statistics across assessed plots including "
        "distribution by risk level, area type, and country."
    ),
    responses={
        200: {"description": "Risk summary retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_risk_summary(
    request: Request,
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    user: AuthUser = Depends(
        require_permission("eudr-pav:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskSummaryResponse:
    """Get risk summary statistics.

    Args:
        country_code: Optional country filter.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskSummaryResponse with aggregate statistics.
    """
    start = time.monotonic()

    try:
        engine = get_risk_scorer()
        result = engine.get_summary(
            country_code=country_code.upper() if country_code else None,
        )

        by_risk_level = [
            RiskSummaryByCategory(
                category=r.get("category", ""),
                count=r.get("count", 0),
                average_risk_score=Decimal(str(r.get("average_risk_score", 0)))
                if r.get("average_risk_score") is not None else None,
                percentage=Decimal(str(r.get("percentage", 0)))
                if r.get("percentage") is not None else None,
            )
            for r in result.get("by_risk_level", [])
        ]

        by_area_type = [
            RiskSummaryByCategory(
                category=r.get("category", ""),
                count=r.get("count", 0),
                average_risk_score=Decimal(str(r.get("average_risk_score", 0)))
                if r.get("average_risk_score") is not None else None,
                percentage=Decimal(str(r.get("percentage", 0)))
                if r.get("percentage") is not None else None,
            )
            for r in result.get("by_area_type", [])
        ]

        by_country = [
            RiskSummaryByCategory(
                category=r.get("category", ""),
                count=r.get("count", 0),
                average_risk_score=Decimal(str(r.get("average_risk_score", 0)))
                if r.get("average_risk_score") is not None else None,
                percentage=Decimal(str(r.get("percentage", 0)))
                if r.get("percentage") is not None else None,
            )
            for r in result.get("by_country", [])
        ]

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"risk_summary:{country_code}",
            str(result.get("total_plots_assessed", 0)),
        )

        logger.info(
            "Risk summary: total=%d critical=%d high=%d operator=%s",
            result.get("total_plots_assessed", 0),
            result.get("critical_plots_count", 0),
            result.get("high_risk_plots_count", 0),
            user.operator_id or user.user_id,
        )

        return RiskSummaryResponse(
            total_plots_assessed=result.get("total_plots_assessed", 0),
            average_risk_score=Decimal(str(result.get("average_risk_score", 0))),
            by_risk_level=by_risk_level,
            by_area_type=by_area_type,
            by_country=by_country,
            critical_plots_count=result.get("critical_plots_count", 0),
            high_risk_plots_count=result.get("high_risk_plots_count", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RiskScorer"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Risk summary retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk summary retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /risk/proximity-alerts
# ---------------------------------------------------------------------------


@router.get(
    "/proximity-alerts",
    response_model=ProximityAlertsResponse,
    summary="Get high-risk proximity alerts",
    description=(
        "Retrieve active high-risk proximity alerts for plots that are "
        "dangerously close to or overlapping with protected areas."
    ),
    responses={
        200: {"description": "Proximity alerts retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_proximity_alerts(
    request: Request,
    risk_level: Optional[RiskLevelEnum] = Query(None, description="Filter by minimum risk level"),
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProximityAlertsResponse:
    """Get high-risk proximity alerts.

    Args:
        risk_level: Optional minimum risk level filter.
        country_code: Optional country filter.
        pagination: Pagination parameters.
        user: Authenticated user with risk:read permission.

    Returns:
        ProximityAlertsResponse with active alerts.
    """
    start = time.monotonic()

    try:
        engine = get_risk_scorer()
        result = engine.get_proximity_alerts(
            risk_level=risk_level.value if risk_level else None,
            country_code=country_code.upper() if country_code else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        alerts = []
        critical_count = 0
        high_count = 0

        for a in result.get("alerts", []):
            level = RiskLevelEnum(a.get("risk_level", "medium"))
            alerts.append(
                ProximityAlertEntry(
                    alert_id=a.get("alert_id", ""),
                    plot_id=a.get("plot_id", ""),
                    area_id=a.get("area_id", ""),
                    area_name=a.get("area_name", ""),
                    distance_km=Decimal(str(a.get("distance_km", 0))),
                    risk_level=level,
                    risk_score=Decimal(str(a.get("risk_score", 0))),
                    alert_reason=a.get("alert_reason", ""),
                )
            )
            if level == RiskLevelEnum.CRITICAL:
                critical_count += 1
            elif level == RiskLevelEnum.HIGH:
                high_count += 1

        total = result.get("total", len(alerts))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"proximity_alerts:{risk_level}:{country_code}", str(total),
        )

        logger.info(
            "Proximity alerts: total=%d critical=%d high=%d operator=%s",
            total,
            critical_count,
            high_count,
            user.operator_id or user.user_id,
        )

        return ProximityAlertsResponse(
            alerts=alerts,
            total_alerts=total,
            critical_count=critical_count,
            high_count=high_count,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["RiskScorer"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Proximity alerts retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Proximity alerts retrieval failed",
        )
