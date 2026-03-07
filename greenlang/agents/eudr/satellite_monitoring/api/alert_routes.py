# -*- coding: utf-8 -*-
"""
Alert Routes - AGENT-EUDR-003 Satellite Monitoring API

Endpoints for satellite monitoring alert management including listing,
detail retrieval, acknowledgement, and summary statistics.

Endpoints:
    GET  /              - List alerts (paginated, filterable)
    GET  /{alert_id}    - Get alert details
    PUT  /{alert_id}/acknowledge - Acknowledge alert
    GET  /summary       - Alert summary statistics

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.satellite_monitoring.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_alert_generator,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    AcknowledgeAlertRequest,
    AlertDetailResponse,
    AlertListResponse,
    AlertSummaryResponse,
    PaginatedMeta,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Satellite Alerts"])


# ---------------------------------------------------------------------------
# In-memory alert store (replaced by database in production)
# ---------------------------------------------------------------------------

_alert_store: Dict[str, Dict[str, Any]] = {}


def _get_alert_store() -> Dict[str, Dict[str, Any]]:
    """Return the alert store. Replaceable for testing."""
    return _alert_store


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=AlertListResponse,
    summary="List satellite monitoring alerts",
    description=(
        "List satellite monitoring alerts with optional filters for "
        "plot_id, severity, and acknowledgement status. Results are "
        "paginated and sorted by creation date descending."
    ),
    responses={
        200: {"description": "Paginated alert list"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_alerts(
    request: Request,
    plot_id: Optional[str] = Query(
        None, description="Filter by plot ID"
    ),
    severity: Optional[str] = Query(
        None, description="Filter by severity: low, medium, high, critical"
    ),
    acknowledged: Optional[bool] = Query(
        None, description="Filter by acknowledgement status"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-satellite:alerts:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
) -> AlertListResponse:
    """List satellite monitoring alerts with filters.

    Args:
        plot_id: Optional plot ID filter.
        severity: Optional severity filter.
        acknowledged: Optional acknowledgement status filter.
        user: Authenticated user with alerts:read permission.
        pagination: Pagination parameters.

    Returns:
        AlertListResponse with paginated alerts.
    """
    logger.info(
        "List alerts: user=%s plot_id=%s severity=%s acknowledged=%s "
        "limit=%d offset=%d",
        user.user_id,
        plot_id,
        severity,
        acknowledged,
        pagination.limit,
        pagination.offset,
    )

    try:
        generator = get_alert_generator()

        result = generator.list_alerts(
            plot_id=plot_id,
            severity=severity,
            acknowledged=acknowledged,
            limit=pagination.limit,
            offset=pagination.offset,
            operator_id=user.operator_id or user.user_id,
        )

        items = getattr(result, "items", [])
        total = getattr(result, "total", len(items))

        alerts = []
        for alert in items:
            alerts.append(AlertDetailResponse(
                alert_id=getattr(alert, "alert_id", ""),
                plot_id=getattr(alert, "plot_id", ""),
                schedule_id=getattr(alert, "schedule_id", None),
                severity=getattr(alert, "severity", "medium"),
                alert_type=getattr(alert, "alert_type", "deforestation"),
                title=getattr(alert, "title", ""),
                description=getattr(alert, "description", ""),
                commodity=getattr(alert, "commodity", ""),
                country_code=getattr(alert, "country_code", ""),
                ndvi_delta=getattr(alert, "ndvi_delta", None),
                forest_loss_ha=getattr(alert, "forest_loss_ha", 0.0),
                confidence=getattr(alert, "confidence", 0.0),
                acknowledged=getattr(alert, "acknowledged", False),
                acknowledged_by=getattr(alert, "acknowledged_by", None),
                acknowledged_at=getattr(alert, "acknowledged_at", None),
                acknowledgement_notes=getattr(alert, "acknowledgement_notes", None),
                detection_date=getattr(alert, "detection_date", datetime.now(timezone.utc).date()),
                data_sources=getattr(alert, "data_sources", []),
                provenance_hash=getattr(alert, "provenance_hash", ""),
                created_at=getattr(alert, "created_at", datetime.now(timezone.utc)),
            ))

        return AlertListResponse(
            alerts=alerts,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as exc:
        logger.error(
            "List alerts failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        return AlertListResponse(
            alerts=[],
            meta=PaginatedMeta(
                total=0,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=False,
            ),
        )


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------

# NOTE: /summary must be defined BEFORE /{alert_id} to avoid FastAPI
# treating "summary" as an alert_id path parameter.


@router.get(
    "/summary",
    response_model=AlertSummaryResponse,
    summary="Get alert summary statistics",
    description=(
        "Retrieve alert summary statistics including counts by severity, "
        "alert type, commodity, and country. Shows total forest loss "
        "and average confidence across all alerts."
    ),
    responses={
        200: {"description": "Alert summary statistics"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_alert_summary(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertSummaryResponse:
    """Get alert summary statistics.

    Returns aggregate alert metrics including counts by severity,
    type, commodity, and country.

    Args:
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertSummaryResponse with summary statistics.
    """
    logger.info(
        "Alert summary: user=%s",
        user.user_id,
    )

    try:
        generator = get_alert_generator()

        summary = generator.get_summary(
            operator_id=user.operator_id or user.user_id,
        )

        if summary is None:
            return AlertSummaryResponse()

        return AlertSummaryResponse(
            total_alerts=getattr(summary, "total_alerts", 0),
            unacknowledged=getattr(summary, "unacknowledged", 0),
            acknowledged=getattr(summary, "acknowledged", 0),
            by_severity=getattr(summary, "by_severity", {
                "critical": 0, "high": 0, "medium": 0, "low": 0,
            }),
            by_alert_type=getattr(summary, "by_alert_type", {
                "deforestation": 0, "degradation": 0, "anomaly": 0, "data_gap": 0,
            }),
            by_commodity=getattr(summary, "by_commodity", {}),
            by_country=getattr(summary, "by_country", {}),
            avg_confidence=getattr(summary, "avg_confidence", 0.0),
            total_forest_loss_ha=getattr(summary, "total_forest_loss_ha", 0.0),
        )

    except Exception as exc:
        logger.error(
            "Alert summary failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        return AlertSummaryResponse()


# ---------------------------------------------------------------------------
# GET /{alert_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{alert_id}",
    response_model=AlertDetailResponse,
    summary="Get alert details",
    description="Retrieve full details for a specific satellite alert.",
    responses={
        200: {"description": "Alert details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def get_alert_detail(
    alert_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertDetailResponse:
    """Get full details for a specific alert.

    Args:
        alert_id: Alert identifier.
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertDetailResponse with full alert data.

    Raises:
        HTTPException: 404 if alert not found.
    """
    logger.info(
        "Alert detail: user=%s alert_id=%s",
        user.user_id,
        alert_id,
    )

    try:
        generator = get_alert_generator()
        alert = generator.get_alert(alert_id=alert_id)

        if alert is None:
            # Check in-memory store as fallback
            store = _get_alert_store()
            alert_data = store.get(alert_id)
            if alert_data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Alert {alert_id} not found",
                )
            alert = type("Alert", (), alert_data)()

        return AlertDetailResponse(
            alert_id=getattr(alert, "alert_id", alert_id),
            plot_id=getattr(alert, "plot_id", ""),
            schedule_id=getattr(alert, "schedule_id", None),
            severity=getattr(alert, "severity", "medium"),
            alert_type=getattr(alert, "alert_type", "deforestation"),
            title=getattr(alert, "title", ""),
            description=getattr(alert, "description", ""),
            commodity=getattr(alert, "commodity", ""),
            country_code=getattr(alert, "country_code", ""),
            ndvi_delta=getattr(alert, "ndvi_delta", None),
            forest_loss_ha=getattr(alert, "forest_loss_ha", 0.0),
            confidence=getattr(alert, "confidence", 0.0),
            acknowledged=getattr(alert, "acknowledged", False),
            acknowledged_by=getattr(alert, "acknowledged_by", None),
            acknowledged_at=getattr(alert, "acknowledged_at", None),
            acknowledgement_notes=getattr(alert, "acknowledgement_notes", None),
            detection_date=getattr(alert, "detection_date", datetime.now(timezone.utc).date()),
            data_sources=getattr(alert, "data_sources", []),
            provenance_hash=getattr(alert, "provenance_hash", ""),
            created_at=getattr(alert, "created_at", datetime.now(timezone.utc)),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Alert detail failed: user=%s alert_id=%s error=%s",
            user.user_id,
            alert_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# PUT /{alert_id}/acknowledge
# ---------------------------------------------------------------------------


@router.put(
    "/{alert_id}/acknowledge",
    response_model=AlertDetailResponse,
    summary="Acknowledge a satellite alert",
    description=(
        "Acknowledge a satellite monitoring alert. Marks the alert as "
        "reviewed with optional notes explaining the action taken."
    ),
    responses={
        200: {"description": "Acknowledged alert"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
        409: {"model": ErrorResponse, "description": "Alert already acknowledged"},
    },
)
async def acknowledge_alert(
    alert_id: str,
    body: AcknowledgeAlertRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:alerts:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AlertDetailResponse:
    """Acknowledge a satellite monitoring alert.

    Args:
        alert_id: Alert identifier.
        body: Acknowledgement request with optional notes.
        user: Authenticated user with alerts:write permission.

    Returns:
        AlertDetailResponse with updated alert.

    Raises:
        HTTPException: 404 if not found, 409 if already acknowledged.
    """
    logger.info(
        "Acknowledge alert: user=%s alert_id=%s",
        user.user_id,
        alert_id,
    )

    try:
        generator = get_alert_generator()
        alert = generator.get_alert(alert_id=alert_id)

        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found",
            )

        if getattr(alert, "acknowledged", False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Alert {alert_id} is already acknowledged",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Acknowledge via the generator engine
        result = generator.acknowledge(
            alert_id=alert_id,
            acknowledged_by=user.user_id,
            notes=body.notes,
        )

        logger.info(
            "Alert acknowledged: alert_id=%s user=%s",
            alert_id,
            user.user_id,
        )

        return AlertDetailResponse(
            alert_id=getattr(result, "alert_id", alert_id),
            plot_id=getattr(result, "plot_id", getattr(alert, "plot_id", "")),
            schedule_id=getattr(result, "schedule_id", getattr(alert, "schedule_id", None)),
            severity=getattr(result, "severity", getattr(alert, "severity", "medium")),
            alert_type=getattr(result, "alert_type", getattr(alert, "alert_type", "deforestation")),
            title=getattr(result, "title", getattr(alert, "title", "")),
            description=getattr(result, "description", getattr(alert, "description", "")),
            commodity=getattr(result, "commodity", getattr(alert, "commodity", "")),
            country_code=getattr(result, "country_code", getattr(alert, "country_code", "")),
            ndvi_delta=getattr(result, "ndvi_delta", getattr(alert, "ndvi_delta", None)),
            forest_loss_ha=getattr(result, "forest_loss_ha", getattr(alert, "forest_loss_ha", 0.0)),
            confidence=getattr(result, "confidence", getattr(alert, "confidence", 0.0)),
            acknowledged=True,
            acknowledged_by=user.user_id,
            acknowledged_at=now,
            acknowledgement_notes=body.notes,
            detection_date=getattr(result, "detection_date", getattr(alert, "detection_date", now.date())),
            data_sources=getattr(result, "data_sources", getattr(alert, "data_sources", [])),
            provenance_hash=getattr(result, "provenance_hash", getattr(alert, "provenance_hash", "")),
            created_at=getattr(result, "created_at", getattr(alert, "created_at", now)),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Alert acknowledgement failed: user=%s alert_id=%s error=%s",
            user.user_id,
            alert_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert acknowledgement failed due to an internal error",
        )
