# -*- coding: utf-8 -*-
"""
Alertmanager Webhook Receiver - OBS-004: Unified Alerting Service

Parses incoming Prometheus Alertmanager webhook payloads and converts
them into the GreenLang Alert model. Designed to be mounted as a
FastAPI route at ``/api/v1/alerts/webhook/alertmanager``.

Reference:
    https://prometheus.io/docs/alerting/latest/configuration/#webhook_config

Example:
    >>> alerts = parse_alertmanager_webhook(payload)
    >>> for alert in alerts:
    ...     await alerting_service.fire_alert(alert.to_dict())

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    Request = None  # type: ignore[assignment, misc]
    JSONResponse = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Alertmanager parser
# ---------------------------------------------------------------------------


def parse_alertmanager_webhook(payload: Dict[str, Any]) -> List[Alert]:
    """Parse an Alertmanager webhook payload into a list of Alerts.

    The Alertmanager webhook format sends a JSON body with:
    - ``status``: group status (``firing`` or ``resolved``)
    - ``alerts``: list of individual alert objects

    Each alert object contains ``status``, ``labels``, ``annotations``,
    ``startsAt``, ``endsAt``, ``generatorURL``, ``fingerprint``.

    Args:
        payload: Raw Alertmanager webhook JSON.

    Returns:
        List of Alert instances.
    """
    am_alerts = payload.get("alerts", [])
    if not am_alerts:
        logger.warning("Empty alerts array in Alertmanager webhook")
        return []

    common_labels = payload.get("commonLabels", {})
    common_annotations = payload.get("commonAnnotations", {})
    external_url = payload.get("externalURL", "")
    group_key = payload.get("groupKey", "")

    alerts: List[Alert] = []
    for am_alert in am_alerts:
        try:
            alert = _parse_alertmanager_alert(
                am_alert,
                common_labels=common_labels,
                common_annotations=common_annotations,
                external_url=external_url,
            )
            alerts.append(alert)
        except Exception as exc:
            logger.error(
                "Failed to parse Alertmanager alert: %s", exc, exc_info=True,
            )

    logger.info(
        "Parsed Alertmanager webhook: %d alerts, group=%s",
        len(alerts), group_key[:30],
    )
    return alerts


def _parse_alertmanager_alert(
    am_alert: Dict[str, Any],
    common_labels: Optional[Dict[str, str]] = None,
    common_annotations: Optional[Dict[str, str]] = None,
    external_url: str = "",
) -> Alert:
    """Convert a single Alertmanager alert dict to a GreenLang Alert.

    Args:
        am_alert: Individual alert from the Alertmanager webhook.
        common_labels: Common labels from the alert group.
        common_annotations: Common annotations from the alert group.
        external_url: Alertmanager external URL.

    Returns:
        Alert instance.
    """
    labels = am_alert.get("labels", {})
    annotations = am_alert.get("annotations", {})

    # Merge common labels/annotations (alert-level takes precedence)
    merged_labels = dict(common_labels or {})
    merged_labels.update(labels)

    merged_annotations = dict(common_annotations or {})
    merged_annotations.update(annotations)

    severity = _map_severity(merged_labels)
    status = _map_status(am_alert.get("status", "firing"))
    fingerprint = _extract_fingerprint(am_alert)

    # Parse timestamps
    fired_at = _parse_iso_timestamp(am_alert.get("startsAt", ""))
    resolved_at = None
    if status == AlertStatus.RESOLVED:
        resolved_at = _parse_iso_timestamp(am_alert.get("endsAt", ""))

    # Extract key fields from labels
    alertname = merged_labels.get("alertname", "unknown")
    job = merged_labels.get("job", "")
    instance = merged_labels.get("instance", "")
    namespace = merged_labels.get("namespace", "")
    service_label = merged_labels.get("service", "")
    team_label = merged_labels.get("team", "")

    # Build title from summary or alertname
    title = merged_annotations.get("summary", alertname)
    description = merged_annotations.get("description", "")

    # Runbook and dashboard links
    runbook_url = merged_annotations.get("runbook_url", "")
    dashboard_url = merged_annotations.get("dashboard_url", "")
    generator_url = am_alert.get("generatorURL", "")

    if not dashboard_url and generator_url:
        dashboard_url = generator_url

    return Alert(
        source="prometheus",
        name=alertname,
        severity=severity,
        status=status,
        title=title,
        description=description,
        fingerprint=fingerprint,
        labels=merged_labels,
        annotations=merged_annotations,
        team=team_label,
        service=service_label or job,
        environment=namespace or merged_labels.get("environment", ""),
        fired_at=fired_at,
        resolved_at=resolved_at,
        runbook_url=runbook_url,
        dashboard_url=dashboard_url,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_severity(labels: Dict[str, str]) -> AlertSeverity:
    """Map Alertmanager severity label to AlertSeverity enum.

    Looks in ``severity`` first, then ``priority``.

    Args:
        labels: Alert labels.

    Returns:
        AlertSeverity value.
    """
    raw = labels.get("severity", labels.get("priority", "warning")).lower()
    mapping = {
        "critical": AlertSeverity.CRITICAL,
        "error": AlertSeverity.CRITICAL,
        "high": AlertSeverity.CRITICAL,
        "warning": AlertSeverity.WARNING,
        "warn": AlertSeverity.WARNING,
        "medium": AlertSeverity.WARNING,
        "info": AlertSeverity.INFO,
        "informational": AlertSeverity.INFO,
        "low": AlertSeverity.INFO,
        "none": AlertSeverity.INFO,
    }
    return mapping.get(raw, AlertSeverity.WARNING)


def _map_status(am_status: str) -> AlertStatus:
    """Map Alertmanager status string to AlertStatus enum.

    Args:
        am_status: ``firing`` or ``resolved``.

    Returns:
        AlertStatus value.
    """
    if am_status.lower() == "resolved":
        return AlertStatus.RESOLVED
    return AlertStatus.FIRING


def _extract_fingerprint(am_alert: Dict[str, Any]) -> str:
    """Extract the dedup fingerprint from an Alertmanager alert.

    Uses the Alertmanager-provided fingerprint if present, otherwise
    generates one from labels.

    Args:
        am_alert: Alert dict.

    Returns:
        Fingerprint string.
    """
    fp = am_alert.get("fingerprint", "")
    if fp:
        return fp

    labels = am_alert.get("labels", {})
    return Alert.generate_fingerprint(
        source="prometheus",
        name=labels.get("alertname", "unknown"),
        labels=labels,
    )


def _parse_iso_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp from Alertmanager.

    Args:
        ts: ISO timestamp string.

    Returns:
        datetime or None.
    """
    if not ts or ts == "0001-01-01T00:00:00Z":
        return None
    try:
        # Handle Go's time format which may have nanoseconds
        cleaned = ts.replace("Z", "+00:00")
        # Truncate nanosecond precision to microsecond
        if "." in cleaned:
            parts = cleaned.split(".")
            frac_and_tz = parts[1]
            # Split fraction from timezone
            for i, c in enumerate(frac_and_tz):
                if c in ("+", "-") and i > 0:
                    frac = frac_and_tz[:i][:6]  # max 6 digits for microseconds
                    tz_part = frac_and_tz[i:]
                    cleaned = f"{parts[0]}.{frac}{tz_part}"
                    break
        return datetime.fromisoformat(cleaned)
    except (ValueError, IndexError) as exc:
        logger.debug("Failed to parse timestamp '%s': %s", ts, exc)
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# FastAPI router (standalone webhook endpoint)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    alertmanager_webhook_router = APIRouter(
        tags=["alerts-webhook"],
    )

    @alertmanager_webhook_router.post(
        "/api/v1/alerts/webhook/alertmanager",
        response_model=None,
        summary="Receive Alertmanager webhook",
    )
    async def receive_alertmanager_webhook(request: Request) -> JSONResponse:
        """Receive and process an Alertmanager webhook payload.

        Parses the webhook body into GreenLang Alert instances and fires
        them through the alerting service if available on ``app.state``.
        """
        payload = await request.json()
        alerts = parse_alertmanager_webhook(payload)

        alerting_service = getattr(request.app.state, "alerting_service", None)
        if alerting_service is not None:
            for alert in alerts:
                try:
                    if alert.status == AlertStatus.RESOLVED:
                        await alerting_service.resolve_alert(
                            alert.alert_id, "alertmanager",
                        )
                    else:
                        await alerting_service.fire_alert(alert.to_dict())
                except Exception as exc:
                    logger.error(
                        "Failed to process AM alert: %s", exc,
                    )

        return JSONResponse(
            status_code=200,
            content={
                "status": "accepted",
                "alerts_processed": len(alerts),
            },
        )
else:
    alertmanager_webhook_router = None  # type: ignore[assignment]
