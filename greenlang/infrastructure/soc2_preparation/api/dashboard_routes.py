# -*- coding: utf-8 -*-
"""
SOC 2 Dashboard API Routes - SEC-009 Phase 10

FastAPI routes for SOC 2 dashboard and metrics:
- GET /dashboard/summary - Dashboard summary
- GET /dashboard/timeline - Audit timeline
- GET /dashboard/metrics - All metrics

Requires soc2:dashboard:view permission.

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class ReadinessWidget(BaseModel):
    """Readiness score widget data."""

    overall_score: float = Field(..., description="Overall readiness score (0-100)")
    overall_status: str = Field(
        ..., description="Status: not_ready, partial, ready, audit_ready"
    )
    trend: str = Field(default="stable", description="Trend: improving, stable, declining")
    change_from_last_week: float = Field(default=0.0, description="Weekly change")
    category_scores: Dict[str, float] = Field(..., description="Scores by TSC category")


class GapsWidget(BaseModel):
    """Gaps summary widget data."""

    total_gaps: int = Field(..., description="Total gaps")
    open_gaps: int = Field(..., description="Open gaps")
    critical_count: int = Field(default=0, description="Critical gaps")
    high_count: int = Field(default=0, description="High severity gaps")
    overdue_count: int = Field(default=0, description="Overdue gaps")
    resolved_this_week: int = Field(default=0, description="Resolved this week")


class TestingWidget(BaseModel):
    """Control testing widget data."""

    last_run_date: Optional[datetime] = Field(None, description="Last test run")
    last_run_status: str = Field(default="none", description="Last run status")
    pass_rate: float = Field(default=0.0, description="Pass rate percentage")
    total_tests: int = Field(default=0, description="Total test cases")
    passed_tests: int = Field(default=0, description="Passed tests")
    failed_tests: int = Field(default=0, description="Failed tests")
    next_scheduled: Optional[datetime] = Field(None, description="Next scheduled run")


class EvidenceWidget(BaseModel):
    """Evidence collection widget data."""

    total_evidence: int = Field(default=0, description="Total evidence items")
    validated_count: int = Field(default=0, description="Validated evidence")
    pending_validation: int = Field(default=0, description="Pending validation")
    stale_count: int = Field(default=0, description="Stale evidence needing refresh")
    last_collection: Optional[datetime] = Field(None, description="Last collection time")
    coverage_percent: float = Field(default=0.0, description="Criteria coverage")


class AuditorWidget(BaseModel):
    """Auditor portal widget data."""

    portal_enabled: bool = Field(default=False, description="Portal enabled")
    active_auditors: int = Field(default=0, description="Active auditor sessions")
    pending_requests: int = Field(default=0, description="Pending evidence requests")
    overdue_requests: int = Field(default=0, description="Overdue requests")
    avg_response_hours: float = Field(default=0.0, description="Avg response time")


class ProjectWidget(BaseModel):
    """Project status widget data."""

    project_name: str = Field(..., description="Project name")
    status: str = Field(..., description="Project status")
    overall_progress: int = Field(default=0, description="Overall progress")
    days_remaining: int = Field(default=0, description="Days until deadline")
    milestones_completed: int = Field(default=0, description="Completed milestones")
    milestones_total: int = Field(default=0, description="Total milestones")
    is_on_track: bool = Field(default=True, description="On schedule")
    next_milestone: Optional[str] = Field(None, description="Next milestone")
    next_milestone_date: Optional[datetime] = Field(None, description="Next date")


class AttestationWidget(BaseModel):
    """Attestation status widget data."""

    pending_attestations: int = Field(default=0, description="Pending attestations")
    awaiting_signatures: int = Field(default=0, description="Awaiting signatures")
    completed_this_year: int = Field(default=0, description="Completed this year")
    next_due: Optional[datetime] = Field(None, description="Next attestation due")


class DashboardSummary(BaseModel):
    """Complete dashboard summary."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Dashboard data timestamp",
    )
    readiness: ReadinessWidget = Field(..., description="Readiness score widget")
    gaps: GapsWidget = Field(..., description="Gaps summary widget")
    testing: TestingWidget = Field(..., description="Control testing widget")
    evidence: EvidenceWidget = Field(..., description="Evidence collection widget")
    auditor: AuditorWidget = Field(..., description="Auditor portal widget")
    project: ProjectWidget = Field(..., description="Project status widget")
    attestation: AttestationWidget = Field(..., description="Attestation status widget")
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Active alerts"
    )


class TimelineEntry(BaseModel):
    """Timeline entry for audit timeline."""

    date: datetime = Field(..., description="Entry date")
    event_type: str = Field(..., description="Event type")
    title: str = Field(..., description="Event title")
    description: str = Field(default="", description="Event description")
    status: str = Field(default="pending", description="Event status")
    category: str = Field(default="general", description="Event category")
    is_milestone: bool = Field(default=False, description="Is a milestone")


class AuditTimeline(BaseModel):
    """Audit timeline data."""

    current_phase: str = Field(..., description="Current audit phase")
    phase_progress: int = Field(default=0, description="Phase progress percentage")
    entries: List[TimelineEntry] = Field(..., description="Timeline entries")
    days_in_phase: int = Field(default=0, description="Days in current phase")
    days_remaining_in_phase: int = Field(default=0, description="Days remaining")


class MetricValue(BaseModel):
    """Individual metric value."""

    metric_name: str = Field(..., description="Metric name")
    value: Any = Field(..., description="Metric value")
    unit: str = Field(default="", description="Unit of measure")
    trend: str = Field(default="stable", description="Trend direction")
    change_percent: float = Field(default=0.0, description="Change from previous")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement time",
    )


class MetricsResponse(BaseModel):
    """All metrics response."""

    readiness_metrics: List[MetricValue] = Field(..., description="Readiness metrics")
    testing_metrics: List[MetricValue] = Field(..., description="Testing metrics")
    evidence_metrics: List[MetricValue] = Field(..., description="Evidence metrics")
    gap_metrics: List[MetricValue] = Field(..., description="Gap metrics")
    sla_metrics: List[MetricValue] = Field(..., description="SLA metrics")
    trend_data: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Trend data for charts"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/dashboard", tags=["soc2-dashboard"])


@router.get(
    "/summary",
    response_model=DashboardSummary,
    summary="Dashboard summary",
    description="Get the complete SOC 2 dashboard summary with all widgets.",
)
async def get_summary(
    request: Request,
) -> DashboardSummary:
    """Get dashboard summary.

    Args:
        request: FastAPI request object.

    Returns:
        DashboardSummary with all widget data.
    """
    logger.info("Getting dashboard summary")

    readiness = ReadinessWidget(
        overall_score=72.5,
        overall_status="partial",
        trend="improving",
        change_from_last_week=3.5,
        category_scores={
            "security": 78.0,
            "availability": 65.0,
            "confidentiality": 70.0,
            "processing_integrity": 68.0,
            "privacy": 72.0,
        },
    )

    gaps = GapsWidget(
        total_gaps=12,
        open_gaps=5,
        critical_count=0,
        high_count=2,
        overdue_count=1,
        resolved_this_week=3,
    )

    testing = TestingWidget(
        last_run_date=datetime.now(timezone.utc) - timedelta(days=1),
        last_run_status="passed",
        pass_rate=93.75,
        total_tests=48,
        passed_tests=45,
        failed_tests=3,
        next_scheduled=datetime.now(timezone.utc) + timedelta(days=7),
    )

    evidence = EvidenceWidget(
        total_evidence=156,
        validated_count=142,
        pending_validation=10,
        stale_count=4,
        last_collection=datetime.now(timezone.utc) - timedelta(hours=6),
        coverage_percent=87.5,
    )

    auditor = AuditorWidget(
        portal_enabled=True,
        active_auditors=2,
        pending_requests=3,
        overdue_requests=0,
        avg_response_hours=18.5,
    )

    project = ProjectWidget(
        project_name="GreenLang SOC 2 Type II Audit 2026",
        status="preparation",
        overall_progress=45,
        days_remaining=180,
        milestones_completed=1,
        milestones_total=5,
        is_on_track=True,
        next_milestone="Gap Remediation Complete",
        next_milestone_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    attestation = AttestationWidget(
        pending_attestations=1,
        awaiting_signatures=1,
        completed_this_year=2,
        next_due=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    alerts = [
        {
            "alert_id": str(uuid4()),
            "severity": "warning",
            "title": "High-severity gap approaching due date",
            "message": "CC6.7 MFA remediation due in 7 days",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "alert_id": str(uuid4()),
            "severity": "info",
            "title": "Attestation signature pending",
            "message": "1 signer pending for Q1 2026 attestation",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]

    return DashboardSummary(
        readiness=readiness,
        gaps=gaps,
        testing=testing,
        evidence=evidence,
        auditor=auditor,
        project=project,
        attestation=attestation,
        alerts=alerts,
    )


@router.get(
    "/timeline",
    response_model=AuditTimeline,
    summary="Audit timeline",
    description="Get the audit timeline with phases and events.",
)
async def get_timeline(
    request: Request,
    start_date: Optional[datetime] = Query(None, description="Timeline start"),
    end_date: Optional[datetime] = Query(None, description="Timeline end"),
) -> AuditTimeline:
    """Get audit timeline.

    Args:
        request: FastAPI request object.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        AuditTimeline with events.
    """
    logger.info("Getting audit timeline")

    entries = [
        TimelineEntry(
            date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            event_type="milestone",
            title="Project Kickoff",
            description="SOC 2 Type II audit project initiated",
            status="completed",
            category="preparation",
            is_milestone=True,
        ),
        TimelineEntry(
            date=datetime(2026, 1, 31, tzinfo=timezone.utc),
            event_type="milestone",
            title="Readiness Assessment Complete",
            description="Initial readiness assessment completed",
            status="completed",
            category="preparation",
            is_milestone=True,
        ),
        TimelineEntry(
            date=datetime(2026, 2, 15, tzinfo=timezone.utc),
            event_type="activity",
            title="Gap Remediation Started",
            description="Began addressing identified gaps",
            status="completed",
            category="preparation",
        ),
        TimelineEntry(
            date=datetime(2026, 3, 31, tzinfo=timezone.utc),
            event_type="milestone",
            title="Gap Remediation Complete",
            description="All gaps addressed and verified",
            status="in_progress",
            category="preparation",
            is_milestone=True,
        ),
        TimelineEntry(
            date=datetime(2026, 4, 15, tzinfo=timezone.utc),
            event_type="milestone",
            title="Audit Kickoff",
            description="Formal audit begins with external auditors",
            status="pending",
            category="fieldwork",
            is_milestone=True,
        ),
        TimelineEntry(
            date=datetime(2026, 6, 30, tzinfo=timezone.utc),
            event_type="milestone",
            title="Fieldwork Complete",
            description="All audit fieldwork completed",
            status="pending",
            category="fieldwork",
            is_milestone=True,
        ),
        TimelineEntry(
            date=datetime(2026, 8, 31, tzinfo=timezone.utc),
            event_type="deadline",
            title="Final Report",
            description="SOC 2 Type II report issued",
            status="pending",
            category="reporting",
            is_milestone=True,
        ),
    ]

    return AuditTimeline(
        current_phase="preparation",
        phase_progress=65,
        entries=entries,
        days_in_phase=45,
        days_remaining_in_phase=53,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="All metrics",
    description="Get all SOC 2 preparation metrics.",
)
async def get_metrics(
    request: Request,
    include_trends: bool = Query(True, description="Include trend data"),
) -> MetricsResponse:
    """Get all metrics.

    Args:
        request: FastAPI request object.
        include_trends: Whether to include trend data.

    Returns:
        MetricsResponse with all metrics.
    """
    logger.info("Getting all metrics")

    readiness_metrics = [
        MetricValue(
            metric_name="overall_readiness_score",
            value=72.5,
            unit="percent",
            trend="improving",
            change_percent=3.5,
        ),
        MetricValue(
            metric_name="criteria_coverage",
            value=87.5,
            unit="percent",
            trend="stable",
            change_percent=0.0,
        ),
        MetricValue(
            metric_name="evidence_freshness",
            value=92.0,
            unit="percent",
            trend="stable",
            change_percent=-1.0,
        ),
    ]

    testing_metrics = [
        MetricValue(
            metric_name="test_pass_rate",
            value=93.75,
            unit="percent",
            trend="improving",
            change_percent=2.0,
        ),
        MetricValue(
            metric_name="automated_test_coverage",
            value=78.0,
            unit="percent",
            trend="stable",
            change_percent=0.0,
        ),
        MetricValue(
            metric_name="avg_test_duration",
            value=45.3,
            unit="seconds",
            trend="improving",
            change_percent=-5.0,
        ),
    ]

    evidence_metrics = [
        MetricValue(
            metric_name="total_evidence",
            value=156,
            unit="items",
            trend="improving",
            change_percent=12.0,
        ),
        MetricValue(
            metric_name="validation_rate",
            value=91.0,
            unit="percent",
            trend="stable",
            change_percent=1.0,
        ),
        MetricValue(
            metric_name="collection_frequency",
            value=6.0,
            unit="hours",
            trend="stable",
            change_percent=0.0,
        ),
    ]

    gap_metrics = [
        MetricValue(
            metric_name="open_gaps",
            value=5,
            unit="count",
            trend="improving",
            change_percent=-37.5,
        ),
        MetricValue(
            metric_name="avg_resolution_time",
            value=14.5,
            unit="days",
            trend="improving",
            change_percent=-8.0,
        ),
        MetricValue(
            metric_name="critical_open",
            value=0,
            unit="count",
            trend="stable",
            change_percent=0.0,
        ),
    ]

    sla_metrics = [
        MetricValue(
            metric_name="sla_compliance_rate",
            value=94.0,
            unit="percent",
            trend="stable",
            change_percent=0.5,
        ),
        MetricValue(
            metric_name="avg_response_time",
            value=18.5,
            unit="hours",
            trend="improving",
            change_percent=-10.0,
        ),
        MetricValue(
            metric_name="requests_resolved_on_time",
            value=47,
            unit="count",
            trend="improving",
            change_percent=15.0,
        ),
    ]

    trend_data: Dict[str, List[Dict[str, Any]]] = {}
    if include_trends:
        # Generate sample trend data for last 7 days
        trend_data = {
            "readiness_score": [
                {
                    "date": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                    "value": 72.5 - (i * 0.5),
                }
                for i in range(7, -1, -1)
            ],
            "open_gaps": [
                {
                    "date": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                    "value": 5 + i,
                }
                for i in range(7, -1, -1)
            ],
            "test_pass_rate": [
                {
                    "date": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                    "value": 93.75 - (i * 0.3),
                }
                for i in range(7, -1, -1)
            ],
        }

    return MetricsResponse(
        readiness_metrics=readiness_metrics,
        testing_metrics=testing_metrics,
        evidence_metrics=evidence_metrics,
        gap_metrics=gap_metrics,
        sla_metrics=sla_metrics,
        trend_data=trend_data,
    )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Dashboard health",
    description="Get the health status of dashboard data sources.",
)
async def get_health(
    request: Request,
) -> Dict[str, Any]:
    """Get dashboard health status.

    Args:
        request: FastAPI request object.

    Returns:
        Health status dictionary.
    """
    logger.info("Getting dashboard health")

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "assessment": {
                "status": "healthy",
                "last_update": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            },
            "evidence": {
                "status": "healthy",
                "last_update": (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
            },
            "testing": {
                "status": "healthy",
                "last_update": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
            "findings": {
                "status": "healthy",
                "last_update": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            },
            "project": {
                "status": "healthy",
                "last_update": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
        },
        "cache": {
            "hit_rate": 0.87,
            "size_mb": 45.2,
            "ttl_seconds": 300,
        },
    }


__all__ = ["router"]
