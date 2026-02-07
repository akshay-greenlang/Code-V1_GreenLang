# -*- coding: utf-8 -*-
"""
Alerting Service REST API Router - OBS-004: Unified Alerting Service

FastAPI router providing CRUD operations on alerts, webhook ingestion,
lifecycle management (acknowledge, resolve, suppress, escalate), on-call
lookups, analytics reports, and channel health checks.

All endpoints are mounted under ``/api/v1/alerts``.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; alerts_router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateAlertRequest(BaseModel):
        """Request body for creating a new alert."""
        source: str = Field(..., description="Alert source system")
        name: str = Field(..., description="Alert rule name")
        severity: str = Field("warning", description="critical, warning, or info")
        title: str = Field(..., description="Human-readable title")
        description: str = Field("", description="Detailed description")
        labels: Dict[str, str] = Field(default_factory=dict)
        annotations: Dict[str, str] = Field(default_factory=dict)
        tenant_id: str = Field("", description="Tenant identifier")
        team: str = Field("", description="Owning team")
        service: str = Field("", description="Affected service")
        environment: str = Field("", description="Environment")
        runbook_url: str = Field("", description="Runbook link")
        dashboard_url: str = Field("", description="Dashboard link")

    class AcknowledgeRequest(BaseModel):
        """Request body for acknowledging an alert."""
        user: str = Field(..., description="User performing acknowledgement")

    class ResolveRequest(BaseModel):
        """Request body for resolving an alert."""
        user: str = Field(..., description="User resolving the alert")

    class SuppressRequest(BaseModel):
        """Request body for suppressing an alert."""
        duration_minutes: int = Field(60, ge=1, le=1440)
        reason: str = Field("", description="Suppression reason")

    class EscalateRequest(BaseModel):
        """Request body for manual escalation."""
        reason: str = Field("", description="Escalation reason")

    class NoteRequest(BaseModel):
        """Request body for adding a note."""
        note: str = Field(..., description="Note text")

    class TestNotificationRequest(BaseModel):
        """Request body for sending a test notification."""
        channel: str = Field("slack", description="Channel to test")
        severity: str = Field("info", description="Test severity")

    class AlertResponse(BaseModel):
        """Serialized alert in API responses."""
        alert_id: str
        fingerprint: str = ""
        source: str = ""
        name: str = ""
        severity: str = ""
        status: str = ""
        title: str = ""
        description: str = ""
        labels: Dict[str, str] = Field(default_factory=dict)
        annotations: Dict[str, str] = Field(default_factory=dict)
        tenant_id: str = ""
        team: str = ""
        service: str = ""
        environment: str = ""
        fired_at: Optional[str] = None
        acknowledged_at: Optional[str] = None
        acknowledged_by: str = ""
        resolved_at: Optional[str] = None
        resolved_by: str = ""
        escalation_level: int = 0
        notification_count: int = 0
        runbook_url: str = ""
        dashboard_url: str = ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_service(request: Request) -> Any:
    """Extract the alerting service from app state.

    Args:
        request: FastAPI request.

    Returns:
        AlertingService instance.

    Raises:
        HTTPException: If not configured.
    """
    svc = getattr(request.app.state, "alerting_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Alerting service not configured",
        )
    return svc


def _alert_to_response(alert: Any) -> Dict[str, Any]:
    """Convert an Alert dataclass to a response dict."""
    d = alert.to_dict()
    return d


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    alerts_router = APIRouter(
        prefix="/api/v1/alerts",
        tags=["alerts"],
    )

    # -- Create alert -------------------------------------------------------

    @alerts_router.post(
        "",
        response_model=AlertResponse,
        status_code=201,
        summary="Create a new alert",
    )
    async def create_alert(
        body: CreateAlertRequest,
        request: Request,
    ) -> JSONResponse:
        """Fire a new alert through the alerting pipeline."""
        svc = _get_service(request)
        alert = await svc.fire_alert(body.model_dump())
        return JSONResponse(
            status_code=201,
            content=_alert_to_response(alert),
        )

    # -- List alerts --------------------------------------------------------

    @alerts_router.get(
        "",
        summary="List alerts",
    )
    async def list_alerts(
        request: Request,
        status: Optional[str] = Query(None, description="Filter by status"),
        severity: Optional[str] = Query(None, description="Filter by severity"),
        team: Optional[str] = Query(None, description="Filter by team"),
        service: Optional[str] = Query(None, description="Filter by service"),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> JSONResponse:
        """List alerts with optional filters."""
        svc = _get_service(request)
        filters: Dict[str, Any] = {}
        if status:
            filters["status"] = status
        if severity:
            filters["severity"] = severity
        if team:
            filters["team"] = team
        if service:
            filters["service"] = service
        filters["limit"] = limit
        filters["offset"] = offset

        alerts = await svc.list_alerts(filters)
        return JSONResponse(
            content={
                "alerts": [_alert_to_response(a) for a in alerts],
                "count": len(alerts),
                "limit": limit,
                "offset": offset,
            }
        )

    # -- Get alert ----------------------------------------------------------

    @alerts_router.get(
        "/{alert_id}",
        summary="Get alert details",
    )
    async def get_alert(
        alert_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a single alert by ID."""
        svc = _get_service(request)
        alert = await svc.get_alert(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        return JSONResponse(content=_alert_to_response(alert))

    # -- Acknowledge --------------------------------------------------------

    @alerts_router.patch(
        "/{alert_id}/acknowledge",
        summary="Acknowledge an alert",
    )
    async def acknowledge_alert(
        alert_id: str,
        body: AcknowledgeRequest,
        request: Request,
    ) -> JSONResponse:
        """Acknowledge an alert."""
        svc = _get_service(request)
        try:
            alert = await svc.acknowledge_alert(alert_id, body.user)
            return JSONResponse(content=_alert_to_response(alert))
        except KeyError:
            raise HTTPException(status_code=404, detail="Alert not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # -- Resolve ------------------------------------------------------------

    @alerts_router.patch(
        "/{alert_id}/resolve",
        summary="Resolve an alert",
    )
    async def resolve_alert(
        alert_id: str,
        body: ResolveRequest,
        request: Request,
    ) -> JSONResponse:
        """Resolve an alert."""
        svc = _get_service(request)
        try:
            alert = await svc.resolve_alert(alert_id, body.user)
            return JSONResponse(content=_alert_to_response(alert))
        except KeyError:
            raise HTTPException(status_code=404, detail="Alert not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # -- Escalate -----------------------------------------------------------

    @alerts_router.patch(
        "/{alert_id}/escalate",
        summary="Manually escalate an alert",
    )
    async def escalate_alert(
        alert_id: str,
        body: EscalateRequest,
        request: Request,
    ) -> JSONResponse:
        """Manually escalate an alert."""
        svc = _get_service(request)
        try:
            alert = await svc.escalate_alert(alert_id, body.reason)
            if alert is None:
                raise HTTPException(status_code=404, detail="Alert not found")
            return JSONResponse(content=_alert_to_response(alert))
        except KeyError:
            raise HTTPException(status_code=404, detail="Alert not found")

    # -- Suppress -----------------------------------------------------------

    @alerts_router.patch(
        "/{alert_id}/suppress",
        summary="Suppress an alert",
    )
    async def suppress_alert(
        alert_id: str,
        body: SuppressRequest,
        request: Request,
    ) -> JSONResponse:
        """Suppress (snooze) an alert."""
        svc = _get_service(request)
        try:
            alert = await svc.suppress_alert(
                alert_id, body.duration_minutes, body.reason,
            )
            return JSONResponse(content=_alert_to_response(alert))
        except KeyError:
            raise HTTPException(status_code=404, detail="Alert not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # -- Add note -----------------------------------------------------------

    @alerts_router.post(
        "/{alert_id}/note",
        summary="Add a note to an alert",
    )
    async def add_note(
        alert_id: str,
        body: NoteRequest,
        request: Request,
    ) -> JSONResponse:
        """Add a note/comment to an alert."""
        svc = _get_service(request)
        alert = await svc.get_alert(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        notes = alert.annotations.get("notes", "")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        alert.annotations["notes"] = f"{notes}\n[{timestamp}] {body.note}".strip()
        return JSONResponse(content=_alert_to_response(alert))

    # -- Alertmanager webhook -----------------------------------------------

    @alerts_router.post(
        "/webhook/alertmanager",
        summary="Alertmanager webhook receiver",
    )
    async def alertmanager_webhook(request: Request) -> JSONResponse:
        """Receive and process an Alertmanager webhook."""
        from greenlang.infrastructure.alerting_service.webhook_receiver import (
            parse_alertmanager_webhook,
        )
        from greenlang.infrastructure.alerting_service.models import AlertStatus

        payload = await request.json()
        alerts = parse_alertmanager_webhook(payload)

        svc = getattr(request.app.state, "alerting_service", None)
        processed = 0
        if svc is not None:
            for alert in alerts:
                try:
                    if alert.status == AlertStatus.RESOLVED:
                        await svc.resolve_alert(alert.alert_id, "alertmanager")
                    else:
                        await svc.fire_alert(alert.to_dict())
                    processed += 1
                except Exception as exc:
                    logger.error("AM webhook alert processing failed: %s", exc)

        return JSONResponse(
            content={"status": "accepted", "alerts_processed": processed},
        )

    # -- Analytics: MTTA ----------------------------------------------------

    @alerts_router.get(
        "/analytics/mtta",
        summary="MTTA report",
    )
    async def analytics_mtta(
        request: Request,
        team: str = Query("", description="Team name"),
        period_hours: int = Query(24, ge=1, le=720),
    ) -> JSONResponse:
        """Get Mean Time To Acknowledge statistics."""
        svc = _get_service(request)
        report = svc.analytics.get_mtta_report(team, period_hours)
        return JSONResponse(content=report)

    # -- Analytics: MTTR ----------------------------------------------------

    @alerts_router.get(
        "/analytics/mttr",
        summary="MTTR report",
    )
    async def analytics_mttr(
        request: Request,
        team: str = Query("", description="Team name"),
        period_hours: int = Query(24, ge=1, le=720),
    ) -> JSONResponse:
        """Get Mean Time To Resolve statistics."""
        svc = _get_service(request)
        report = svc.analytics.get_mttr_report(team, period_hours)
        return JSONResponse(content=report)

    # -- Analytics: Fatigue -------------------------------------------------

    @alerts_router.get(
        "/analytics/fatigue",
        summary="Alert fatigue report",
    )
    async def analytics_fatigue(
        request: Request,
        team: str = Query("", description="Team name"),
    ) -> JSONResponse:
        """Get alert fatigue score (alerts per hour)."""
        svc = _get_service(request)
        score = svc.analytics.get_fatigue_score(team)
        return JSONResponse(content={"team": team or "all", "fatigue_score": score})

    # -- Analytics: Top noisy -----------------------------------------------

    @alerts_router.get(
        "/analytics/top-noisy",
        summary="Top noisy alerts",
    )
    async def analytics_top_noisy(
        request: Request,
        limit: int = Query(10, ge=1, le=100),
    ) -> JSONResponse:
        """Get most frequently firing alert names."""
        svc = _get_service(request)
        results = svc.analytics.get_top_noisy_alerts(limit)
        return JSONResponse(content={"alerts": results})

    # -- On-call: All schedules ---------------------------------------------

    @alerts_router.get(
        "/oncall",
        summary="List on-call schedules",
    )
    async def list_oncall(request: Request) -> JSONResponse:
        """List known on-call schedules."""
        svc = _get_service(request)
        schedules = await svc.oncall.list_schedules()
        return JSONResponse(
            content={
                "schedules": [
                    {
                        "schedule_id": s.schedule_id,
                        "name": s.name,
                        "provider": s.provider,
                        "current_oncall": {
                            "user_id": s.current_oncall.user_id,
                            "name": s.current_oncall.name,
                            "email": s.current_oncall.email,
                        }
                        if s.current_oncall
                        else None,
                    }
                    for s in schedules
                ]
            }
        )

    # -- On-call: Specific schedule -----------------------------------------

    @alerts_router.get(
        "/oncall/{schedule_id}",
        summary="Get on-call schedule",
    )
    async def get_oncall_schedule(
        schedule_id: str,
        request: Request,
        provider: str = Query("", description="pagerduty or opsgenie"),
    ) -> JSONResponse:
        """Get current on-call for a specific schedule."""
        svc = _get_service(request)
        schedule = await svc.oncall.get_oncall_schedule(
            schedule_id, provider,
        )
        if schedule is None:
            raise HTTPException(
                status_code=404, detail="Schedule not found",
            )
        return JSONResponse(
            content={
                "schedule_id": schedule.schedule_id,
                "name": schedule.name,
                "provider": schedule.provider,
                "timezone": schedule.timezone,
                "current_oncall": {
                    "user_id": schedule.current_oncall.user_id,
                    "name": schedule.current_oncall.name,
                    "email": schedule.current_oncall.email,
                    "phone": schedule.current_oncall.phone,
                }
                if schedule.current_oncall
                else None,
            }
        )

    # -- Channel health -----------------------------------------------------

    @alerts_router.get(
        "/channels/health",
        summary="Channel health check",
    )
    async def channels_health(request: Request) -> JSONResponse:
        """Check health of all registered notification channels."""
        svc = _get_service(request)
        all_channels = svc.channel_registry.list_channels()
        healthy = await svc.channel_registry.get_healthy_channels()
        return JSONResponse(
            content={
                "channels": {
                    ch: {"healthy": ch in healthy}
                    for ch in all_channels
                },
                "total": len(all_channels),
                "healthy": len(healthy),
            }
        )

    # -- Test notification --------------------------------------------------

    @alerts_router.post(
        "/test",
        summary="Send test notification",
    )
    async def send_test_notification(
        body: TestNotificationRequest,
        request: Request,
    ) -> JSONResponse:
        """Send a test notification through the specified channel."""
        svc = _get_service(request)

        from greenlang.infrastructure.alerting_service.models import (
            Alert,
            AlertSeverity,
        )

        test_alert = Alert(
            source="test",
            name="TestAlert",
            severity=AlertSeverity(body.severity),
            title="Test Notification from GreenLang Alerting",
            description="This is a test notification to verify channel connectivity.",
            team="platform",
            service="alerting-service",
            environment=svc.config.environment,
        )

        results = await svc.router.notify(
            test_alert, [body.channel], svc.template_engine,
        )
        return JSONResponse(
            content={
                "channel": body.channel,
                "results": [
                    {
                        "status": r.status.value,
                        "duration_ms": r.duration_ms,
                        "error_message": r.error_message,
                    }
                    for r in results
                ],
            }
        )

else:
    alerts_router = None  # type: ignore[assignment]
