# -*- coding: utf-8 -*-
"""
Historical Baseline Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for historical baseline management using 2018-2020 reference period
canopy cover and forest area measurements with minimum 3-sample requirement
and 10% canopy cover threshold for forest classification.

Endpoints:
    POST /baseline/establish         - Establish new baseline for a plot
    POST /baseline/compare           - Compare current state to baseline
    PUT  /baseline/{baseline_id}     - Update existing baseline
    GET  /baseline/coverage          - Baseline coverage statistics

Reference Period: 2018-2020
Minimum Samples: 3
Canopy Cover Threshold: 10%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, HistoricalBaselineEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_baseline_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    BaselineCompareRequest,
    BaselineCompareResponse,
    BaselineCoverageEntry,
    BaselineCoverageResponse,
    BaselineDataEntry,
    BaselineEstablishRequest,
    BaselineEstablishResponse,
    BaselineUpdateRequest,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    SatelliteSourceEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/baseline", tags=["Historical Baselines"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /baseline/establish
# ---------------------------------------------------------------------------


@router.post(
    "/establish",
    response_model=BaselineEstablishResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Establish historical baseline for a plot",
    description=(
        "Establish a historical forest cover baseline for a supply chain plot "
        "using satellite observations from the reference period (default 2018-2020). "
        "Requires minimum 3 observations to establish a reliable baseline."
    ),
    responses={
        201: {"description": "Baseline established"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Insufficient baseline samples"},
    },
)
async def establish_baseline(
    request: Request,
    body: BaselineEstablishRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:baseline:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BaselineEstablishResponse:
    """Establish a historical baseline for a plot.

    Args:
        body: Baseline establishment request.
        user: Authenticated user with baseline:create permission.

    Returns:
        BaselineEstablishResponse with baseline data.
    """
    start = time.monotonic()

    try:
        engine = get_baseline_engine()
        result = engine.establish(
            plot_id=body.plot_id,
            latitude=float(body.center.latitude),
            longitude=float(body.center.longitude),
            radius_km=float(body.radius_km) if body.radius_km else None,
            polygon=[
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.polygon.coordinates
            ] if body.polygon else None,
            reference_start_year=body.reference_start_year,
            reference_end_year=body.reference_end_year,
            sources=[s.value for s in body.sources] if body.sources else None,
            min_samples=body.min_samples,
        )

        baseline_data = []
        for entry in result.get("baseline_data", []):
            baseline_data.append(
                BaselineDataEntry(
                    observation_date=entry.get("observation_date"),
                    canopy_cover_pct=Decimal(str(entry.get("canopy_cover_pct", 0))),
                    forest_area_ha=Decimal(str(entry.get("forest_area_ha", 0)))
                    if entry.get("forest_area_ha") is not None else None,
                    ndvi_mean=Decimal(str(entry.get("ndvi_mean", 0)))
                    if entry.get("ndvi_mean") is not None else None,
                    source=SatelliteSourceEnum(entry.get("source", "sentinel2")),
                )
            )

        avg_cover = Decimal(str(result.get("average_canopy_cover_pct", 0)))
        is_forested = avg_cover >= Decimal("10")
        ref_period = f"{body.reference_start_year}-{body.reference_end_year}"

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"establish:{body.plot_id}:{ref_period}",
            str(avg_cover),
        )

        logger.info(
            "Baseline established: plot_id=%s samples=%d avg_cover=%s is_forested=%s operator=%s",
            body.plot_id,
            len(baseline_data),
            avg_cover,
            is_forested,
            user.operator_id or user.user_id,
        )

        return BaselineEstablishResponse(
            baseline_id=result.get("baseline_id", ""),
            plot_id=body.plot_id,
            reference_period=ref_period,
            sample_count=len(baseline_data),
            average_canopy_cover_pct=avg_cover,
            average_forest_area_ha=Decimal(str(result.get("average_forest_area_ha", 0)))
            if result.get("average_forest_area_ha") is not None else None,
            baseline_data=baseline_data,
            is_forested=is_forested,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=list({d.source.value for d in baseline_data}),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Baseline establishment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical baseline establishment failed",
        )


# ---------------------------------------------------------------------------
# POST /baseline/compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=BaselineCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare current state to historical baseline",
    description=(
        "Compare the current forest cover state against an established "
        "historical baseline to detect significant deforestation changes."
    ),
    responses={
        200: {"description": "Comparison completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Baseline not found"},
    },
)
async def compare_baseline(
    request: Request,
    body: BaselineCompareRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:baseline:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BaselineCompareResponse:
    """Compare current state against a historical baseline.

    Args:
        body: Comparison request.
        user: Authenticated user with baseline:read permission.

    Returns:
        BaselineCompareResponse with comparison results.
    """
    start = time.monotonic()

    try:
        engine = get_baseline_engine()
        result = engine.compare(
            baseline_id=body.baseline_id,
            current_observation_date=body.current_observation_date,
            include_timeline=body.include_timeline,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Baseline not found: {body.baseline_id}",
            )

        change_pct = Decimal(str(result.get("change_pct", 0)))
        deforestation_detected = change_pct < Decimal("-5")

        significance = "not_significant"
        if abs(change_pct) >= Decimal("20"):
            significance = "significant"
        elif abs(change_pct) >= Decimal("10"):
            significance = "moderate"

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compare:{body.baseline_id}",
            str(change_pct),
        )

        logger.info(
            "Baseline comparison: baseline_id=%s change_pct=%s deforestation=%s operator=%s",
            body.baseline_id,
            change_pct,
            deforestation_detected,
            user.operator_id or user.user_id,
        )

        return BaselineCompareResponse(
            baseline_id=body.baseline_id,
            plot_id=result.get("plot_id", ""),
            baseline_canopy_cover_pct=Decimal(str(result.get("baseline_canopy_cover_pct", 0))),
            current_canopy_cover_pct=Decimal(str(result.get("current_canopy_cover_pct", 0))),
            change_pct=change_pct,
            change_area_ha=Decimal(str(result.get("change_area_ha", 0)))
            if result.get("change_area_ha") is not None else None,
            deforestation_detected=deforestation_detected,
            change_significance=significance,
            comparison_date=result.get("comparison_date", date.today()),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["HistoricalBaselineEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Baseline comparison failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline comparison failed",
        )


# ---------------------------------------------------------------------------
# PUT /baseline/{baseline_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{baseline_id}",
    response_model=BaselineEstablishResponse,
    summary="Update existing baseline",
    description=(
        "Update an existing baseline by adding new samples or adjusting "
        "the reference period."
    ),
    responses={
        200: {"description": "Baseline updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Baseline not found"},
    },
)
async def update_baseline(
    baseline_id: str,
    request: Request,
    body: BaselineUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:baseline:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BaselineEstablishResponse:
    """Update an existing baseline.

    Args:
        baseline_id: Baseline to update.
        body: Update request.
        user: Authenticated user with baseline:update permission.

    Returns:
        BaselineEstablishResponse with updated baseline.
    """
    start = time.monotonic()

    try:
        engine = get_baseline_engine()
        add_samples_data = None
        if body.add_samples:
            add_samples_data = [
                {
                    "observation_date": s.observation_date,
                    "canopy_cover_pct": float(s.canopy_cover_pct),
                    "forest_area_ha": float(s.forest_area_ha) if s.forest_area_ha else None,
                    "ndvi_mean": float(s.ndvi_mean) if s.ndvi_mean else None,
                    "source": s.source.value,
                }
                for s in body.add_samples
            ]

        result = engine.update_baseline(
            baseline_id=baseline_id,
            add_samples=add_samples_data,
            reference_start_year=body.reference_start_year,
            reference_end_year=body.reference_end_year,
            notes=body.notes,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Baseline not found: {baseline_id}",
            )

        avg_cover = Decimal(str(result.get("average_canopy_cover_pct", 0)))
        is_forested = avg_cover >= Decimal("10")

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"update_baseline:{baseline_id}",
            str(avg_cover),
        )

        logger.info(
            "Baseline updated: baseline_id=%s samples=%d operator=%s",
            baseline_id,
            result.get("sample_count", 0),
            user.operator_id or user.user_id,
        )

        return BaselineEstablishResponse(
            baseline_id=baseline_id,
            plot_id=result.get("plot_id", ""),
            reference_period=result.get("reference_period", "2018-2020"),
            sample_count=result.get("sample_count", 0),
            average_canopy_cover_pct=avg_cover,
            average_forest_area_ha=Decimal(str(result.get("average_forest_area_ha", 0)))
            if result.get("average_forest_area_ha") is not None else None,
            is_forested=is_forested,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["HistoricalBaselineEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Baseline update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline update failed",
        )


# ---------------------------------------------------------------------------
# GET /baseline/coverage
# ---------------------------------------------------------------------------


@router.get(
    "/coverage",
    response_model=BaselineCoverageResponse,
    summary="Get baseline coverage statistics",
    description=(
        "Retrieve coverage statistics for all baselines showing which plots "
        "have adequate baseline data and which need additional samples."
    ),
    responses={
        200: {"description": "Coverage statistics retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_coverage(
    request: Request,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:baseline:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BaselineCoverageResponse:
    """Get baseline coverage statistics.

    Args:
        pagination: Pagination parameters.
        user: Authenticated user with baseline:read permission.

    Returns:
        BaselineCoverageResponse with coverage data.
    """
    start = time.monotonic()

    try:
        engine = get_baseline_engine()
        result = engine.get_coverage(
            limit=pagination.limit,
            offset=pagination.offset,
        )

        baselines = []
        for entry in result.get("baselines", []):
            baselines.append(
                BaselineCoverageEntry(
                    baseline_id=entry.get("baseline_id", ""),
                    plot_id=entry.get("plot_id", ""),
                    reference_period=entry.get("reference_period", "2018-2020"),
                    sample_count=entry.get("sample_count", 0),
                    is_adequate=entry.get("is_adequate", False),
                    average_canopy_cover_pct=Decimal(str(entry.get("average_canopy_cover_pct", 0))),
                    last_updated=entry.get("last_updated"),
                )
            )

        total = result.get("total_baselines", len(baselines))
        adequate = result.get("adequate_coverage_count", 0)
        inadequate = result.get("inadequate_coverage_count", 0)
        coverage_rate = Decimal(str(adequate / total)) if total > 0 else None

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "baseline_coverage", str(total)
        )

        logger.info(
            "Baseline coverage retrieved: total=%d adequate=%d operator=%s",
            total,
            adequate,
            user.operator_id or user.user_id,
        )

        return BaselineCoverageResponse(
            baselines=baselines,
            total_baselines=total,
            adequate_coverage_count=adequate,
            inadequate_coverage_count=inadequate,
            coverage_rate=coverage_rate.quantize(Decimal("0.01"))
            if coverage_rate is not None else None,
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
                data_sources=["HistoricalBaselineEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Baseline coverage retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline coverage retrieval failed",
        )
