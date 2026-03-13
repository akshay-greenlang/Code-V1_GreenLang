# -*- coding: utf-8 -*-
"""
Alert Management Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for deforestation alert lifecycle management including creation,
listing with filters, detail retrieval, batch creation from detections,
summary statistics by country/severity, and alert trend analysis.

Endpoints:
    GET  /alerts               - List alerts with filters
    GET  /alerts/{alert_id}    - Get alert detail
    POST /alerts               - Create manual alert
    POST /alerts/batch         - Batch create from detections
    GET  /alerts/summary       - Alert summary by country/severity
    GET  /alerts/statistics    - Alert statistics and trends

Alert lifecycle: NEW -> TRIAGED -> ASSIGNED -> INVESTIGATING -> RESOLVED/ESCALATED
Retention: 5 years per EUDR Article 31

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, AlertGenerator Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_alert_generator,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
    validate_date_range,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    AlertBatchRequest,
    AlertBatchResponse,
    AlertCreateRequest,
    AlertDetailResponse,
    AlertEntry,
    AlertListResponse,
    AlertSeverityEnum,
    AlertStatusEnum,
    AlertStatisticsResponse,
    AlertSummaryByCategory,
    AlertSummaryResponse,
    ChangeTypeEnum,
    ErrorResponse,
    EUDRCommodityEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert Management"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GET /alerts
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List deforestation alerts with filters",
    description=(
        "Retrieve a paginated list of deforestation alerts with optional filters "
        "for severity, status, country, commodity, date range, and change type. "
        "Results are ordered by creation date descending."
    ),
    responses={
        200: {"description": "Alerts listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_alerts(
    request: Request,
    severity: Optional[AlertSeverityEnum] = Query(None, description="Filter by severity"),
    alert_status: Optional[AlertStatusEnum] = Query(None, alias="status", description="Filter by status"),
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    commodity: Optional[EUDRCommodityEnum] = Query(None, description="Filter by commodity"),
    change_type: Optional[ChangeTypeEnum] = Query(None, description="Filter by change type"),
    date_range: Dict[str, Optional[date]] = Depends(validate_date_range),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertListResponse:
    """List deforestation alerts with optional filters.

    Args:
        severity: Optional severity filter.
        alert_status: Optional status filter.
        country_code: Optional country filter.
        commodity: Optional commodity filter.
        change_type: Optional change type filter.
        date_range: Optional date range filter.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        AlertListResponse with paginated alert list.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.list_alerts(
            severity=severity.value if severity else None,
            status=alert_status.value if alert_status else None,
            country_code=validate_country_code(country_code) if country_code else None,
            commodity=commodity.value if commodity else None,
            change_type=change_type.value if change_type else None,
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
            limit=pagination.limit,
            offset=pagination.offset,
        )

        alerts = []
        for entry in result.get("alerts", []):
            alerts.append(
                AlertEntry(
                    alert_id=entry.get("alert_id", ""),
                    detection_id=entry.get("detection_id"),
                    latitude=Decimal(str(entry.get("latitude", 0))),
                    longitude=Decimal(str(entry.get("longitude", 0))),
                    area_ha=Decimal(str(entry.get("area_ha", 0))),
                    change_type=ChangeTypeEnum(entry.get("change_type", "deforestation")),
                    severity=AlertSeverityEnum(entry.get("severity", "medium")),
                    status=AlertStatusEnum(entry.get("status", "new")),
                    country_code=entry.get("country_code"),
                    confidence=Decimal(str(entry.get("confidence", 0)))
                    if entry.get("confidence") is not None else None,
                    created_at=entry.get("created_at"),
                    updated_at=entry.get("updated_at"),
                )
            )

        total = result.get("total", len(alerts))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_alerts:{severity}:{alert_status}:{country_code}",
            str(total),
        )

        logger.info(
            "Alerts listed: total=%d returned=%d operator=%s",
            total,
            len(alerts),
            user.operator_id or user.user_id,
        )

        return AlertListResponse(
            alerts=alerts,
            total_alerts=total,
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
                data_sources=["DeforestationAlertSystem"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert listing failed",
        )


# ---------------------------------------------------------------------------
# GET /alerts/{alert_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{alert_id}",
    response_model=AlertDetailResponse,
    summary="Get alert detail",
    description=(
        "Retrieve detailed information for a specific deforestation alert "
        "including associated detection data, severity scoring breakdown, "
        "affected plots, workflow history, and cutoff status."
    ),
    responses={
        200: {"description": "Alert detail retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def get_alert_detail(
    alert_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertDetailResponse:
    """Get detailed information for a specific alert.

    Args:
        alert_id: Unique alert identifier.
        user: Authenticated user.

    Returns:
        AlertDetailResponse with full alert details.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.get_alert(alert_id=alert_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )

        alert_data = result.get("alert", {})
        alert_entry = AlertEntry(
            alert_id=alert_data.get("alert_id", alert_id),
            detection_id=alert_data.get("detection_id"),
            latitude=Decimal(str(alert_data.get("latitude", 0))),
            longitude=Decimal(str(alert_data.get("longitude", 0))),
            area_ha=Decimal(str(alert_data.get("area_ha", 0))),
            change_type=ChangeTypeEnum(alert_data.get("change_type", "deforestation")),
            severity=AlertSeverityEnum(alert_data.get("severity", "medium")),
            status=AlertStatusEnum(alert_data.get("status", "new")),
            country_code=alert_data.get("country_code"),
            confidence=Decimal(str(alert_data.get("confidence", 0)))
            if alert_data.get("confidence") is not None else None,
            created_at=alert_data.get("created_at"),
            updated_at=alert_data.get("updated_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"alert_detail:{alert_id}",
            str(alert_entry.severity.value),
        )

        logger.info(
            "Alert detail retrieved: alert_id=%s severity=%s operator=%s",
            alert_id,
            alert_entry.severity.value,
            user.operator_id or user.user_id,
        )

        return AlertDetailResponse(
            alert=alert_entry,
            detection=result.get("detection"),
            severity_score=result.get("severity_score"),
            affected_plot_ids=result.get("affected_plot_ids", []),
            commodities=[
                EUDRCommodityEnum(c) for c in result.get("commodities", [])
            ],
            workflow_history=result.get("workflow_history"),
            cutoff_status=result.get("cutoff_status"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystem"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /alerts
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=AlertDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create manual deforestation alert",
    description=(
        "Manually create a deforestation alert from field observations or "
        "external intelligence. Optionally links to an existing satellite "
        "detection and auto-classifies severity."
    ),
    responses={
        201: {"description": "Alert created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_alert(
    request: Request,
    body: AlertCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AlertDetailResponse:
    """Create a manual deforestation alert.

    Args:
        body: Alert creation request.
        user: Authenticated user with alerts:create permission.

    Returns:
        AlertDetailResponse with created alert.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.create_alert(
            detection_id=body.detection_id,
            latitude=float(body.latitude),
            longitude=float(body.longitude),
            area_ha=float(body.area_ha),
            change_type=body.change_type.value,
            severity=body.severity.value if body.severity else None,
            country_code=body.country_code,
            description=body.description,
            source=body.source.value if body.source else None,
            confidence=float(body.confidence) if body.confidence else None,
            affected_plot_ids=body.affected_plot_ids,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            created_by=user.user_id,
        )

        alert_data = result.get("alert", {})
        alert_entry = AlertEntry(
            alert_id=alert_data.get("alert_id", ""),
            detection_id=alert_data.get("detection_id"),
            latitude=Decimal(str(alert_data.get("latitude", body.latitude))),
            longitude=Decimal(str(alert_data.get("longitude", body.longitude))),
            area_ha=Decimal(str(alert_data.get("area_ha", body.area_ha))),
            change_type=ChangeTypeEnum(alert_data.get("change_type", body.change_type.value)),
            severity=AlertSeverityEnum(alert_data.get("severity", "medium")),
            status=AlertStatusEnum(alert_data.get("status", "new")),
            country_code=alert_data.get("country_code", body.country_code),
            confidence=Decimal(str(alert_data.get("confidence", 0)))
            if alert_data.get("confidence") is not None else body.confidence,
            created_at=alert_data.get("created_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"create_alert:{body.latitude},{body.longitude}:{body.area_ha}",
            alert_entry.alert_id,
        )

        logger.info(
            "Alert created: alert_id=%s severity=%s operator=%s",
            alert_entry.alert_id,
            alert_entry.severity.value,
            user.operator_id or user.user_id,
        )

        return AlertDetailResponse(
            alert=alert_entry,
            affected_plot_ids=body.affected_plot_ids or [],
            commodities=[EUDRCommodityEnum(c) for c in body.commodities] if body.commodities else [],
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ManualCreation"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert creation failed",
        )


# ---------------------------------------------------------------------------
# POST /alerts/batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=AlertBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch create alerts from satellite detections",
    description=(
        "Create alerts in batch from a list of satellite detection IDs. "
        "Performs deduplication to prevent duplicate alerts for the same event. "
        "Optionally auto-classifies severity and performs triage."
    ),
    responses={
        200: {"description": "Batch alert creation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_create_alerts(
    request: Request,
    body: AlertBatchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> AlertBatchResponse:
    """Batch create alerts from satellite detections.

    Args:
        body: Batch request with detection IDs and options.
        user: Authenticated user with alerts:create permission.

    Returns:
        AlertBatchResponse with creation results.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.batch_create_alerts(
            detection_ids=body.detection_ids,
            auto_classify_severity=body.auto_classify_severity,
            auto_triage=body.auto_triage,
            created_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"batch_create:{len(body.detection_ids)}",
            f"{result.get('alerts_created', 0)}/{result.get('alerts_deduplicated', 0)}",
        )

        logger.info(
            "Batch alert creation: requested=%d created=%d dedup=%d failed=%d operator=%s",
            len(body.detection_ids),
            result.get("alerts_created", 0),
            result.get("alerts_deduplicated", 0),
            result.get("alerts_failed", 0),
            user.operator_id or user.user_id,
        )

        return AlertBatchResponse(
            alerts_created=result.get("alerts_created", 0),
            alerts_deduplicated=result.get("alerts_deduplicated", 0),
            alerts_failed=result.get("alerts_failed", 0),
            alert_ids=result.get("alert_ids", []),
            errors=result.get("errors"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystem"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch alert creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch alert creation failed",
        )


# ---------------------------------------------------------------------------
# GET /alerts/summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=AlertSummaryResponse,
    summary="Get alert summary by country and severity",
    description=(
        "Retrieve aggregated alert counts grouped by severity, country, "
        "status, and commodity for dashboard overview."
    ),
    responses={
        200: {"description": "Alert summary retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_alert_summary(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertSummaryResponse:
    """Get alert summary statistics.

    Args:
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertSummaryResponse with grouped counts.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.get_alert_summary()

        def _build_categories(data: List[Dict]) -> List[AlertSummaryByCategory]:
            return [
                AlertSummaryByCategory(
                    category=d.get("category", ""),
                    count=d.get("count", 0),
                    percentage=Decimal(str(d.get("percentage", 0)))
                    if d.get("percentage") is not None else None,
                )
                for d in data
            ]

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "alert_summary",
            str(result.get("total_alerts", 0)),
        )

        logger.info(
            "Alert summary retrieved: total=%d operator=%s",
            result.get("total_alerts", 0),
            user.operator_id or user.user_id,
        )

        return AlertSummaryResponse(
            total_alerts=result.get("total_alerts", 0),
            by_severity=_build_categories(result.get("by_severity", [])),
            by_country=_build_categories(result.get("by_country", [])),
            by_status=_build_categories(result.get("by_status", [])),
            by_commodity=_build_categories(result.get("by_commodity", [])),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystem"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert summary retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert summary retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /alerts/statistics
# ---------------------------------------------------------------------------


@router.get(
    "/statistics",
    response_model=AlertStatisticsResponse,
    summary="Get alert statistics and trends",
    description=(
        "Retrieve detailed alert statistics including active/resolved counts, "
        "false positive rates, average resolution time, SLA compliance, "
        "and recent alert volume trends."
    ),
    responses={
        200: {"description": "Alert statistics retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_alert_statistics(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertStatisticsResponse:
    """Get detailed alert statistics and trends.

    Args:
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertStatisticsResponse with statistics.
    """
    start = time.monotonic()

    try:
        engine = get_alert_generator()
        result = engine.get_alert_statistics()

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "alert_statistics",
            str(result.get("total_alerts", 0)),
        )

        logger.info(
            "Alert statistics retrieved: total=%d active=%d operator=%s",
            result.get("total_alerts", 0),
            result.get("active_alerts", 0),
            user.operator_id or user.user_id,
        )

        return AlertStatisticsResponse(
            total_alerts=result.get("total_alerts", 0),
            active_alerts=result.get("active_alerts", 0),
            resolved_alerts=result.get("resolved_alerts", 0),
            false_positives=result.get("false_positives", 0),
            false_positive_rate=Decimal(str(result.get("false_positive_rate", 0)))
            if result.get("false_positive_rate") is not None else None,
            average_resolution_hours=Decimal(str(result.get("average_resolution_hours", 0)))
            if result.get("average_resolution_hours") is not None else None,
            total_affected_area_ha=Decimal(str(result.get("total_affected_area_ha", 0))),
            alerts_last_24h=result.get("alerts_last_24h", 0),
            alerts_last_7d=result.get("alerts_last_7d", 0),
            alerts_last_30d=result.get("alerts_last_30d", 0),
            sla_compliance_rate=Decimal(str(result.get("sla_compliance_rate", 0)))
            if result.get("sla_compliance_rate") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystem"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert statistics retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert statistics retrieval failed",
        )
