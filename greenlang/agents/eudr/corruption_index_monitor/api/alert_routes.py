# -*- coding: utf-8 -*-
"""
Alert Management Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for corruption index alert management including listing alerts
with filters, retrieving alert details, configuring alert rules,
acknowledging alerts, and viewing alert summary statistics.

Endpoints:
    GET  /alerts                        - List alerts with filters
    GET  /alerts/{alert_id}             - Get alert detail
    POST /alerts/configure              - Configure alert rule
    POST /alerts/{alert_id}/acknowledge - Acknowledge alert
    GET  /alerts/summary                - Alert summary statistics

Alert Types: cpi_change, wgi_change, trend_reversal, threshold_breach,
             country_reclassification, bribery_risk_escalation,
             institutional_degradation
Severity Levels: low, medium, high, critical

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, Alert Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.corruption_index_monitor.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_alert_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    AlertAcknowledgeRequest,
    AlertAcknowledgeResponse,
    AlertConfigRequest,
    AlertConfigResponse,
    AlertDetailResponse,
    AlertEntry,
    AlertListResponse,
    AlertSeverityCount,
    AlertSeverityEnum,
    AlertStatusEnum,
    AlertSummaryResponse,
    AlertTypeEnum,
    AlertTypeSummary,
    ErrorResponse as SchemaErrorResponse,
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
    summary="List corruption index alerts",
    description=(
        "Retrieve a paginated list of corruption index alerts with optional "
        "filters by severity, type, status, and country code. Returns alerts "
        "sorted by creation time descending (newest first)."
    ),
    responses={
        200: {"description": "Alert list retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_alerts(
    request: Request,
    severity: Optional[AlertSeverityEnum] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertTypeEnum] = Query(None, description="Filter by alert type"),
    alert_status: Optional[AlertStatusEnum] = Query(None, description="Filter by status"),
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertListResponse:
    """List corruption index alerts with optional filters.

    Args:
        severity: Optional severity filter.
        alert_type: Optional alert type filter.
        alert_status: Optional status filter.
        country_code: Optional country code filter.
        pagination: Pagination parameters.
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertListResponse with paginated alerts.
    """
    start = time.monotonic()

    try:
        engine = get_alert_engine()
        result = engine.list_alerts(
            severity=severity.value if severity else None,
            alert_type=alert_type.value if alert_type else None,
            status=alert_status.value if alert_status else None,
            country_code=country_code.upper() if country_code else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        alerts = []
        for alert_data in result.get("alerts", []):
            alerts.append(
                AlertEntry(
                    alert_id=alert_data.get("alert_id", ""),
                    alert_type=AlertTypeEnum(alert_data.get("alert_type", "cpi_change")),
                    severity=AlertSeverityEnum(alert_data.get("severity", "medium")),
                    status=AlertStatusEnum(alert_data.get("status", "active")),
                    country_code=alert_data.get("country_code", ""),
                    country_name=alert_data.get("country_name", ""),
                    title=alert_data.get("title", ""),
                    description=alert_data.get("description", ""),
                    index_type=alert_data.get("index_type", "cpi"),
                    previous_value=Decimal(str(alert_data.get("previous_value", 0))) if alert_data.get("previous_value") is not None else None,
                    current_value=Decimal(str(alert_data.get("current_value", 0))) if alert_data.get("current_value") is not None else None,
                    change_magnitude=Decimal(str(alert_data.get("change_magnitude", 0))) if alert_data.get("change_magnitude") is not None else None,
                    threshold_breached=Decimal(str(alert_data.get("threshold_breached", 0))) if alert_data.get("threshold_breached") is not None else None,
                    recommended_actions=alert_data.get("recommended_actions", []),
                )
            )

        total = result.get("total_alerts", len(alerts))
        active_count = result.get("active_count", 0)
        critical_count = result.get("critical_count", 0)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"alerts:list:{severity}:{alert_type}:{alert_status}",
            str(total),
        )

        logger.info(
            "Alerts listed: total=%d active=%d critical=%d operator=%s",
            total,
            active_count,
            critical_count,
            user.operator_id or user.user_id,
        )

        return AlertListResponse(
            alerts=alerts,
            total_alerts=total,
            active_count=active_count,
            critical_count=critical_count,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            metadata=MetadataSchema(),
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
        "Retrieve full details for a specific alert including country context, "
        "related alerts, and historical alert count."
    ),
    responses={
        200: {"description": "Alert detail retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Alert not found"},
    },
)
async def get_alert_detail(
    alert_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertDetailResponse:
    """Get full detail for a specific alert.

    Args:
        alert_id: Unique alert identifier.
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertDetailResponse with full alert data and context.
    """
    start = time.monotonic()

    try:
        engine = get_alert_engine()
        result = engine.get_alert(alert_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found",
            )

        alert_data = result.get("alert", {})
        alert = AlertEntry(
            alert_id=alert_data.get("alert_id", alert_id),
            alert_type=AlertTypeEnum(alert_data.get("alert_type", "cpi_change")),
            severity=AlertSeverityEnum(alert_data.get("severity", "medium")),
            status=AlertStatusEnum(alert_data.get("status", "active")),
            country_code=alert_data.get("country_code", ""),
            country_name=alert_data.get("country_name", ""),
            title=alert_data.get("title", ""),
            description=alert_data.get("description", ""),
            index_type=alert_data.get("index_type", "cpi"),
            previous_value=Decimal(str(alert_data.get("previous_value", 0))) if alert_data.get("previous_value") is not None else None,
            current_value=Decimal(str(alert_data.get("current_value", 0))) if alert_data.get("current_value") is not None else None,
            change_magnitude=Decimal(str(alert_data.get("change_magnitude", 0))) if alert_data.get("change_magnitude") is not None else None,
            recommended_actions=alert_data.get("recommended_actions", []),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"alert_detail:{alert_id}", str(alert.severity.value)
        )

        logger.info(
            "Alert detail retrieved: id=%s type=%s severity=%s operator=%s",
            alert_id,
            alert.alert_type.value,
            alert.severity.value,
            user.operator_id or user.user_id,
        )

        return AlertDetailResponse(
            alert=alert,
            related_alerts=result.get("related_alerts", []),
            country_context=result.get("country_context", {}),
            historical_alerts=result.get("historical_alerts", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(),
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
# POST /alerts/configure
# ---------------------------------------------------------------------------


@router.post(
    "/configure",
    response_model=AlertConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Configure an alert rule",
    description=(
        "Create or update an alert rule configuration specifying type, "
        "severity, threshold, country scope, cooldown period, and "
        "notification channels."
    ),
    responses={
        201: {"description": "Alert rule configured"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def configure_alert(
    request: Request,
    body: AlertConfigRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:alerts:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AlertConfigResponse:
    """Configure an alert rule.

    Args:
        body: Alert configuration request.
        user: Authenticated user with alerts:create permission.

    Returns:
        AlertConfigResponse with saved configuration.
    """
    start = time.monotonic()

    try:
        engine = get_alert_engine()
        result = engine.configure_alert(
            alert_type=body.alert_type.value,
            enabled=body.enabled,
            severity_override=body.severity_override.value if body.severity_override else None,
            threshold_value=float(body.threshold_value) if body.threshold_value else None,
            country_codes=[cc.upper() for cc in body.country_codes] if body.country_codes else None,
            cooldown_hours=body.cooldown_hours,
            notification_channels=body.notification_channels,
        )

        country_scope = "all countries"
        if body.country_codes:
            country_scope = f"{len(body.country_codes)} countries: {', '.join(body.country_codes[:5])}"
            if len(body.country_codes) > 5:
                country_scope += f" (+{len(body.country_codes) - 5} more)"

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"alert_config:{body.alert_type.value}", str(body.enabled)
        )

        logger.info(
            "Alert configured: type=%s enabled=%s scope=%s operator=%s",
            body.alert_type.value,
            body.enabled,
            country_scope,
            user.operator_id or user.user_id,
        )

        return AlertConfigResponse(
            config_id=result.get("config_id", ""),
            alert_type=body.alert_type,
            enabled=body.enabled,
            severity=body.severity_override,
            threshold_value=body.threshold_value,
            country_scope=country_scope,
            cooldown_hours=body.cooldown_hours,
            notification_channels=body.notification_channels,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert configuration failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert configuration failed",
        )


# ---------------------------------------------------------------------------
# POST /alerts/{alert_id}/acknowledge
# ---------------------------------------------------------------------------


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AlertAcknowledgeResponse,
    summary="Acknowledge an alert",
    description=(
        "Acknowledge an active alert, optionally resolving it. Records the "
        "acknowledging user, timestamp, notes, and action taken."
    ),
    responses={
        200: {"description": "Alert acknowledged"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Alert not found"},
    },
)
async def acknowledge_alert(
    alert_id: str,
    request: Request,
    body: AlertAcknowledgeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:alerts:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AlertAcknowledgeResponse:
    """Acknowledge an alert.

    Args:
        alert_id: Unique alert identifier.
        body: Acknowledgement request with notes and action.
        user: Authenticated user with alerts:update permission.

    Returns:
        AlertAcknowledgeResponse with updated status.
    """
    start = time.monotonic()

    try:
        engine = get_alert_engine()
        result = engine.acknowledge_alert(
            alert_id=alert_id,
            user_id=user.user_id,
            notes=body.notes,
            action_taken=body.action_taken,
            resolve=body.resolve,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found",
            )

        new_status = AlertStatusEnum.RESOLVED if body.resolve else AlertStatusEnum.ACKNOWLEDGED

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"ack:{alert_id}:{user.user_id}", str(new_status.value)
        )

        logger.info(
            "Alert acknowledged: id=%s status=%s resolved=%s operator=%s",
            alert_id,
            new_status.value,
            body.resolve,
            user.operator_id or user.user_id,
        )

        return AlertAcknowledgeResponse(
            alert_id=alert_id,
            status=new_status,
            acknowledged_by=user.user_id,
            resolved=body.resolve,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert acknowledgement failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert acknowledgement failed",
        )


# ---------------------------------------------------------------------------
# GET /alerts/summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=AlertSummaryResponse,
    summary="Get alert summary statistics",
    description=(
        "Retrieve summary statistics for all alerts including counts by "
        "severity and type, top affected countries, and recent alert volumes."
    ),
    responses={
        200: {"description": "Alert summary retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_alert_summary(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AlertSummaryResponse:
    """Get alert summary statistics.

    Args:
        user: Authenticated user with alerts:read permission.

    Returns:
        AlertSummaryResponse with aggregated alert statistics.
    """
    start = time.monotonic()

    try:
        engine = get_alert_engine()
        result = engine.get_summary()

        severity_breakdown = []
        for sev_data in result.get("severity_breakdown", []):
            severity_breakdown.append(
                AlertSeverityCount(
                    severity=AlertSeverityEnum(sev_data.get("severity", "medium")),
                    count=sev_data.get("count", 0),
                )
            )

        type_breakdown = []
        for type_data in result.get("type_breakdown", []):
            type_breakdown.append(
                AlertTypeSummary(
                    alert_type=AlertTypeEnum(type_data.get("alert_type", "cpi_change")),
                    count=type_data.get("count", 0),
                    latest=type_data.get("latest"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "alert_summary", str(result.get("total_active", 0))
        )

        logger.info(
            "Alert summary retrieved: active=%d critical=%d operator=%s",
            result.get("total_active", 0),
            sum(s.count for s in severity_breakdown if s.severity == AlertSeverityEnum.CRITICAL),
            user.operator_id or user.user_id,
        )

        return AlertSummaryResponse(
            total_active=result.get("total_active", 0),
            total_acknowledged=result.get("total_acknowledged", 0),
            total_resolved=result.get("total_resolved", 0),
            severity_breakdown=severity_breakdown,
            type_breakdown=type_breakdown,
            top_affected_countries=result.get("top_affected_countries", []),
            alerts_last_24h=result.get("alerts_last_24h", 0),
            alerts_last_7d=result.get("alerts_last_7d", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert summary retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert summary retrieval failed",
        )
