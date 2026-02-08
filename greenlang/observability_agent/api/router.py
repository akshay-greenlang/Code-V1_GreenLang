# -*- coding: utf-8 -*-
"""
Observability Agent Service REST API Router - AGENT-FOUND-010: Observability Agent

FastAPI router providing 20 endpoints for metric recording, distributed tracing,
structured log aggregation, alert evaluation, health checking, dashboard
provisioning, SLO tracking, and service health monitoring.

All endpoints are mounted under ``/api/v1/observability-agent``.

Endpoints:
    1.  POST   /v1/metrics/record           - Record a metric observation
    2.  GET    /v1/metrics                   - List metric definitions
    3.  GET    /v1/metrics/{metric_name}     - Get metric details and current value
    4.  POST   /v1/metrics/export            - Export metrics in Prometheus format
    5.  POST   /v1/traces/spans              - Create/start a trace span
    6.  PUT    /v1/traces/spans/{span_id}    - End/update a trace span
    7.  GET    /v1/traces/{trace_id}         - Get full trace with all spans
    8.  POST   /v1/logs                      - Ingest structured log entries
    9.  GET    /v1/logs                      - Query log entries (with filters)
    10. POST   /v1/alerts/rules              - Create/update alert rule
    11. GET    /v1/alerts/rules              - List alert rules
    12. POST   /v1/alerts/evaluate           - Evaluate all alert rules
    13. GET    /v1/alerts/active             - Get currently firing alerts
    14. POST   /v1/health/check              - Run health check probes
    15. GET    /v1/health/status             - Get aggregated health status
    16. GET    /v1/dashboards/{dashboard_id} - Get dashboard data
    17. POST   /v1/slos                      - Create/update SLO definition
    18. GET    /v1/slos                      - List SLO definitions with compliance
    19. GET    /v1/slos/{slo_id}/burn-rate   - Get SLO burn rate analysis
    20. GET    /health                       - Service health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; observability agent router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RecordMetricBody(BaseModel):
        """Request body for recording a metric observation."""
        metric_name: str = Field(..., description="Name of the metric to record")
        value: float = Field(..., description="Metric observation value")
        labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
        tenant_id: str = Field(default="default", description="Tenant identifier")
        metric_type: str = Field(
            default="gauge",
            description="Metric type (counter, gauge, histogram, summary)",
        )
        description: str = Field(default="", description="Metric description")
        unit: str = Field(default="", description="Metric unit")

    class ExportMetricsBody(BaseModel):
        """Request body for exporting metrics."""
        format: str = Field(default="prometheus", description="Export format")
        metric_names: List[str] = Field(
            default_factory=list,
            description="Filter by metric names (empty = all)",
        )
        tenant_id: Optional[str] = Field(None, description="Filter by tenant")

    class CreateSpanBody(BaseModel):
        """Request body for creating/starting a trace span."""
        operation_name: str = Field(..., description="Name of the operation")
        trace_id: Optional[str] = Field(
            None, description="Trace ID (auto-generated if not provided)",
        )
        parent_span_id: Optional[str] = Field(
            None, description="Parent span ID for nested spans",
        )
        service_name: str = Field(default="unknown", description="Service name")
        attributes: Dict[str, Any] = Field(
            default_factory=dict, description="Span attributes",
        )

    class EndSpanBody(BaseModel):
        """Request body for ending/updating a trace span."""
        status: str = Field(
            default="ok", description="Span status (ok, error, timeout, cancelled)",
        )
        attributes: Dict[str, Any] = Field(
            default_factory=dict, description="Additional attributes to set on close",
        )

    class IngestLogBody(BaseModel):
        """Request body for ingesting structured log entries."""
        level: str = Field(
            default="info",
            description="Log level (debug, info, warning, error, critical)",
        )
        message: str = Field(..., description="Log message")
        agent_id: str = Field(default="", description="Originating agent ID")
        tenant_id: str = Field(default="default", description="Tenant identifier")
        trace_id: Optional[str] = Field(
            None, description="Correlated trace ID",
        )
        span_id: Optional[str] = Field(
            None, description="Correlated span ID",
        )
        correlation_id: Optional[str] = Field(
            None, description="Business correlation ID",
        )
        attributes: Dict[str, Any] = Field(
            default_factory=dict, description="Structured log attributes",
        )

    class CreateAlertRuleBody(BaseModel):
        """Request body for creating/updating an alert rule."""
        name: str = Field(..., description="Alert rule name")
        metric_name: str = Field(..., description="Metric name to evaluate")
        condition: str = Field(
            ..., description="Condition operator (gt, lt, gte, lte, eq, ne)",
        )
        threshold: float = Field(..., description="Threshold value for the condition")
        duration_seconds: int = Field(
            default=60, ge=0, description="Duration in seconds before firing",
        )
        severity: str = Field(
            default="warning",
            description="Alert severity (info, warning, error, critical)",
        )
        labels: Dict[str, str] = Field(
            default_factory=dict, description="Alert labels",
        )
        annotations: Dict[str, str] = Field(
            default_factory=dict,
            description="Alert annotations (summary, description, runbook)",
        )

    class RunHealthCheckBody(BaseModel):
        """Request body for running health check probes."""
        probe_type: Optional[str] = Field(
            None, description="Probe type (liveness, readiness, startup) or None for all",
        )
        service_filter: Optional[str] = Field(
            None, description="Filter checks by service name pattern",
        )

    class CreateSLOBody(BaseModel):
        """Request body for creating/updating an SLO definition."""
        name: str = Field(..., description="SLO name")
        description: str = Field(default="", description="SLO description")
        service_name: str = Field(..., description="Target service name")
        slo_type: str = Field(
            ...,
            description="SLO type (availability, latency, throughput, error_rate)",
        )
        target: float = Field(
            ..., ge=0.0, le=1.0, description="SLO target ratio (e.g. 0.999)",
        )
        window_days: int = Field(
            default=30, ge=1, le=365, description="Rolling window in days",
        )
        burn_rate_thresholds: Dict[str, float] = Field(
            default_factory=lambda: {
                "fast": 14.4,
                "medium": 6.0,
                "slow": 1.0,
            },
            description="Burn rate thresholds by speed tier",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/observability-agent",
        tags=["observability-agent"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract ObservabilityAgentService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        ObservabilityAgentService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "observability_agent_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Observability agent service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Record a metric observation
    @router.post("/v1/metrics/record")
    async def record_metric(
        body: RecordMetricBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Record a metric observation.

        Accepts a metric name, value, labels, and tenant ID. Returns
        the recorded metric entry with provenance hash.
        """
        service = _get_service(request)
        try:
            result = service.record_metric(
                name=body.metric_name,
                value=body.value,
                labels=body.labels,
                tenant_id=body.tenant_id,
                metric_type=body.metric_type,
                description=body.description,
                unit=body.unit,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error recording metric: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 2. List metric definitions
    @router.get("/v1/metrics")
    async def list_metrics(
        tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
        metric_type: Optional[str] = Query(None, description="Filter by type"),
        limit: int = Query(100, ge=1, le=1000, description="Max results"),
        offset: int = Query(0, ge=0, description="Result offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List metric definitions.

        Returns all registered metric definitions with optional
        filtering by tenant and metric type.
        """
        service = _get_service(request)
        try:
            metrics = service.metrics_collector.list_metrics(
                tenant_id=tenant_id,
                metric_type=metric_type,
            )
            total = len(metrics)
            page = metrics[offset : offset + limit]
            return {
                "metrics": [
                    m.model_dump(mode="json") if hasattr(m, "model_dump") else m
                    for m in page
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("Error listing metrics: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 3. Get metric details and current value
    @router.get("/v1/metrics/{metric_name}")
    async def get_metric(
        metric_name: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get metric details and current value.

        Returns the full metric definition including its latest
        observation value and labels.
        """
        service = _get_service(request)
        try:
            metric = service.metrics_collector.get_metric(metric_name)
            if metric is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Metric '{metric_name}' not found",
                )
            return metric.model_dump(mode="json") if hasattr(metric, "model_dump") else metric
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error getting metric %s: %s", metric_name, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 4. Export metrics in Prometheus format
    @router.post("/v1/metrics/export")
    async def export_metrics(
        body: ExportMetricsBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Export metrics in Prometheus exposition format.

        Returns metrics formatted for Prometheus scraping or other
        monitoring backends.
        """
        service = _get_service(request)
        try:
            export_data = service.metrics_collector.export_metrics(
                format=body.format,
                metric_names=body.metric_names or None,
                tenant_id=body.tenant_id,
            )
            return {
                "format": body.format,
                "data": export_data,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error exporting metrics: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 5. Create/start a trace span
    @router.post("/v1/traces/spans")
    async def create_span(
        body: CreateSpanBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create and start a new trace span.

        Initiates a span within a trace. If no trace_id is given, a
        new trace is created. Returns the span record with IDs.
        """
        service = _get_service(request)
        try:
            result = service.start_span(
                name=body.operation_name,
                trace_id=body.trace_id,
                parent_span_id=body.parent_span_id,
                service_name=body.service_name,
                attributes=body.attributes,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error creating span: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 6. End/update a trace span
    @router.put("/v1/traces/spans/{span_id}")
    async def end_span(
        span_id: str,
        body: EndSpanBody,
        request: Request,
    ) -> Dict[str, Any]:
        """End or update a trace span.

        Marks the span as completed with a final status and optional
        additional attributes.
        """
        service = _get_service(request)
        try:
            result = service.end_span(
                span_id=span_id,
                status=body.status,
                attributes=body.attributes,
            )
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Span '{span_id}' not found",
                )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error ending span %s: %s", span_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 7. Get full trace with all spans
    @router.get("/v1/traces/{trace_id}")
    async def get_trace(
        trace_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a full trace with all spans.

        Returns the complete trace tree including all child spans,
        timing, and attributes.
        """
        service = _get_service(request)
        try:
            trace = service.trace_manager.get_trace(trace_id)
            if trace is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trace '{trace_id}' not found",
                )
            return trace.model_dump(mode="json") if hasattr(trace, "model_dump") else trace
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error getting trace %s: %s", trace_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 8. Ingest structured log entries
    @router.post("/v1/logs")
    async def ingest_log(
        body: IngestLogBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Ingest a structured log entry.

        Accepts a structured log entry with level, message, agent ID,
        and optional trace correlation. Returns the stored log record.
        """
        service = _get_service(request)
        try:
            result = service.log(
                level=body.level,
                message=body.message,
                agent_id=body.agent_id,
                tenant_id=body.tenant_id,
                trace_id=body.trace_id,
                span_id=body.span_id,
                correlation_id=body.correlation_id,
                attributes=body.attributes,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error ingesting log: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 9. Query log entries
    @router.get("/v1/logs")
    async def query_logs(
        level: Optional[str] = Query(None, description="Filter by log level"),
        agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
        tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
        trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
        correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
        limit: int = Query(100, ge=1, le=1000, description="Max results"),
        offset: int = Query(0, ge=0, description="Result offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Query log entries with filters.

        Returns structured log entries matching the provided filter
        criteria, ordered by timestamp descending.
        """
        service = _get_service(request)
        try:
            logs = service.log_aggregator.query_logs(
                level=level,
                agent_id=agent_id,
                tenant_id=tenant_id,
                trace_id=trace_id,
                correlation_id=correlation_id,
            )
            total = len(logs)
            page = logs[offset : offset + limit]
            return {
                "logs": [
                    entry.model_dump(mode="json") if hasattr(entry, "model_dump") else entry
                    for entry in page
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("Error querying logs: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 10. Create/update alert rule
    @router.post("/v1/alerts/rules")
    async def create_alert_rule(
        body: CreateAlertRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create or update an alert rule.

        Registers an alert rule that is evaluated periodically against
        the specified metric and condition.
        """
        service = _get_service(request)
        try:
            result = service.add_alert_rule(
                name=body.name,
                metric_name=body.metric_name,
                condition=body.condition,
                threshold=body.threshold,
                duration_seconds=body.duration_seconds,
                severity=body.severity,
                labels=body.labels,
                annotations=body.annotations,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error creating alert rule: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 11. List alert rules
    @router.get("/v1/alerts/rules")
    async def list_alert_rules(
        severity: Optional[str] = Query(None, description="Filter by severity"),
        limit: int = Query(100, ge=1, le=500, description="Max results"),
        offset: int = Query(0, ge=0, description="Result offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List alert rules.

        Returns all registered alert rules with optional severity
        filtering.
        """
        service = _get_service(request)
        try:
            rules = service.alert_evaluator.list_rules(severity=severity)
            total = len(rules)
            page = rules[offset : offset + limit]
            return {
                "rules": [
                    r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                    for r in page
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("Error listing alert rules: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 12. Evaluate all alert rules
    @router.post("/v1/alerts/evaluate")
    async def evaluate_alerts(
        request: Request,
    ) -> Dict[str, Any]:
        """Evaluate all alert rules.

        Triggers an immediate evaluation of every registered alert
        rule and returns the evaluation summary.
        """
        service = _get_service(request)
        try:
            result = service.evaluate_alerts()
            return result
        except Exception as exc:
            logger.error("Error evaluating alerts: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 13. Get currently firing alerts
    @router.get("/v1/alerts/active")
    async def get_active_alerts(
        severity: Optional[str] = Query(None, description="Filter by severity"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get currently firing alerts.

        Returns all alerts that are in firing state, optionally
        filtered by severity level.
        """
        service = _get_service(request)
        try:
            alerts = service.alert_evaluator.get_active_alerts(severity=severity)
            return {
                "alerts": [
                    a.model_dump(mode="json") if hasattr(a, "model_dump") else a
                    for a in alerts
                ],
                "count": len(alerts),
            }
        except Exception as exc:
            logger.error("Error getting active alerts: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 14. Run health check probes
    @router.post("/v1/health/check")
    async def run_health_check(
        body: RunHealthCheckBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run health check probes.

        Executes health check probes (liveness, readiness, startup)
        against registered services and returns the results.
        """
        service = _get_service(request)
        try:
            result = service.check_health(
                probe_type=body.probe_type,
                service_filter=body.service_filter,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error running health check: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 15. Get aggregated health status
    @router.get("/v1/health/status")
    async def get_health_status(
        request: Request,
    ) -> Dict[str, Any]:
        """Get aggregated health status.

        Returns the overall health status of all monitored services
        including individual service health states.
        """
        service = _get_service(request)
        try:
            status = service.health_checker.get_status()
            return status.model_dump(mode="json") if hasattr(status, "model_dump") else status
        except Exception as exc:
            logger.error("Error getting health status: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 16. Get dashboard data
    @router.get("/v1/dashboards/{dashboard_id}")
    async def get_dashboard(
        dashboard_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get dashboard data.

        Returns pre-computed dashboard data for the specified
        dashboard ID including panels, queries, and time ranges.
        """
        service = _get_service(request)
        try:
            data = service.get_dashboard_data(dashboard_id)
            if data is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dashboard '{dashboard_id}' not found",
                )
            return data
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Error getting dashboard %s: %s", dashboard_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # 17. Create/update SLO definition
    @router.post("/v1/slos")
    async def create_slo(
        body: CreateSLOBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create or update an SLO definition.

        Registers a Service Level Objective with target, evaluation
        window, and burn rate thresholds.
        """
        service = _get_service(request)
        try:
            result = service.create_slo(
                name=body.name,
                description=body.description,
                service_name=body.service_name,
                slo_type=body.slo_type,
                target=body.target,
                window_days=body.window_days,
                burn_rate_thresholds=body.burn_rate_thresholds,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Error creating SLO: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 18. List SLO definitions with compliance
    @router.get("/v1/slos")
    async def list_slos(
        service_name: Optional[str] = Query(None, description="Filter by service"),
        slo_type: Optional[str] = Query(None, description="Filter by SLO type"),
        limit: int = Query(100, ge=1, le=500, description="Max results"),
        offset: int = Query(0, ge=0, description="Result offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List SLO definitions with current compliance.

        Returns all registered SLO definitions including their current
        compliance ratios and error budget status.
        """
        service = _get_service(request)
        try:
            slos = service.slo_tracker.list_slos(
                service_name=service_name,
                slo_type=slo_type,
            )
            total = len(slos)
            page = slos[offset : offset + limit]
            return {
                "slos": [
                    s.model_dump(mode="json") if hasattr(s, "model_dump") else s
                    for s in page
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("Error listing SLOs: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # 19. Get SLO burn rate analysis
    @router.get("/v1/slos/{slo_id}/burn-rate")
    async def get_slo_burn_rate(
        slo_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get SLO burn rate analysis.

        Returns the burn rate analysis for a specific SLO including
        fast, medium, and slow burn rate windows and whether the
        error budget is being consumed at an unsustainable rate.
        """
        service = _get_service(request)
        try:
            result = service.slo_tracker.get_burn_rate(slo_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"SLO '{slo_id}' not found",
                )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Error getting burn rate for SLO %s: %s", slo_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # 20. Service health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Observability agent service health check endpoint.

        Lightweight endpoint for load balancer and orchestrator
        health probes.
        """
        return {"status": "healthy", "service": "observability-agent"}


__all__ = [
    "router",
]
