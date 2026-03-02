"""
Dashboard API Routes for GL-EUDR-APP v1.0

Provides aggregated KPIs, compliance trend data, and recent alerts
for the EUDR compliance dashboard. All metrics are computed in
real-time from in-memory stores (v1.0); production would use
materialized views or pre-computed aggregates.

Prefix: /api/v1/dashboard
Tags: Dashboard
"""

import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskDistribution(BaseModel):
    """Distribution of entities across risk levels."""

    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0
    unknown: int = 0


class DashboardMetrics(BaseModel):
    """Top-level dashboard KPIs.

    Example response::

        {
            "total_suppliers": 245,
            "compliant_suppliers": 180,
            "compliant_percent": 73.5,
            "non_compliant_suppliers": 25,
            "pending_suppliers": 40,
            "total_plots": 1230,
            "validated_plots": 1100,
            "total_dds": 312,
            "dds_submitted": 280,
            "dds_pending": 32,
            "total_documents": 890,
            "documents_verified": 750,
            "risk_distribution": {"low": 80, "medium": 100, "high": 50, "critical": 15},
            "active_alerts": 12,
            "pipeline_runs_today": 8,
            "last_updated": "2025-11-09T10:30:00Z"
        }
    """

    total_suppliers: int = Field(0, description="Total registered suppliers")
    compliant_suppliers: int = Field(0, description="Suppliers with compliant status")
    compliant_percent: float = Field(0.0, description="Compliance rate percentage")
    non_compliant_suppliers: int = Field(0, description="Non-compliant suppliers")
    pending_suppliers: int = Field(0, description="Suppliers with pending status")
    total_plots: int = Field(0, description="Total registered plots")
    validated_plots: int = Field(0, description="Plots with validated geometry")
    total_dds: int = Field(0, description="Total Due Diligence Statements")
    dds_submitted: int = Field(0, description="DDS submitted to EU system")
    dds_pending: int = Field(0, description="DDS in draft or pending status")
    total_documents: int = Field(0, description="Total supporting documents")
    documents_verified: int = Field(0, description="Documents verified")
    risk_distribution: RiskDistribution = Field(
        default_factory=RiskDistribution,
        description="Supplier distribution by risk level",
    )
    active_alerts: int = Field(0, description="Currently active risk alerts")
    pipeline_runs_today: int = Field(0, description="Pipeline runs executed today")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Metrics computation timestamp",
    )


class MonthlyTrend(BaseModel):
    """Monthly compliance trend data point."""

    month: str = Field(..., description="Month in YYYY-MM format")
    total_suppliers: int
    compliant_count: int
    compliance_rate: float = Field(..., ge=0, le=100)
    dds_submitted: int
    new_alerts: int
    average_risk_score: float


class ComplianceTrendResponse(BaseModel):
    """Compliance trend data for the last 12 months."""

    months: int = Field(12, description="Number of months covered")
    data: List[MonthlyTrend]
    overall_trend: str = Field(
        ..., description="improving | stable | declining"
    )


class DashboardAlert(BaseModel):
    """A recent alert for the dashboard."""

    alert_id: str
    severity: str = Field(..., description="info | warning | critical")
    title: str
    description: str
    category: str = Field(
        ..., description="deforestation | compliance | document | regulatory"
    )
    created_at: datetime
    acknowledged: bool = False


class DashboardAlertListResponse(BaseModel):
    """List of recent dashboard alerts."""

    items: List[DashboardAlert]
    total: int
    unacknowledged: int


# ---------------------------------------------------------------------------
# Import In-Memory Stores (lazy, to avoid circular imports at module level)
# ---------------------------------------------------------------------------


def _get_supplier_store() -> Dict:
    """Lazy import of supplier storage."""
    try:
        from .supplier_routes import _suppliers
        return _suppliers
    except ImportError:
        return {}


def _get_plot_store() -> Dict:
    """Lazy import of plot storage."""
    try:
        from .plot_routes import _plots
        return _plots
    except ImportError:
        return {}


def _get_dds_store() -> Dict:
    """Lazy import of DDS storage."""
    try:
        from .dds_routes import _dds_store
        return _dds_store
    except ImportError:
        return {}


def _get_document_store() -> Dict:
    """Lazy import of document storage."""
    try:
        from .document_routes import _documents
        return _documents
    except ImportError:
        return {}


def _get_pipeline_store() -> Dict:
    """Lazy import of pipeline storage."""
    try:
        from .pipeline_routes import _pipeline_runs
        return _pipeline_runs
    except ImportError:
        return {}


def _get_alert_store() -> List:
    """Lazy import of alert storage."""
    try:
        from .risk_routes import _alerts, _generate_seed_alerts
        _generate_seed_alerts()
        return _alerts
    except ImportError:
        return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    response_model=DashboardMetrics,
    summary="Dashboard KPIs",
    description="Retrieve top-level dashboard metrics and KPIs.",
)
async def get_dashboard_metrics() -> DashboardMetrics:
    """
    Compute and return real-time dashboard KPIs aggregated across
    all in-memory stores.

    Returns:
        200 with current metrics.
    """
    suppliers = _get_supplier_store()
    plots = _get_plot_store()
    dds_store = _get_dds_store()
    documents = _get_document_store()
    pipelines = _get_pipeline_store()
    alerts = _get_alert_store()

    # Supplier metrics
    total_suppliers = len(suppliers)
    compliant = sum(
        1 for s in suppliers.values() if s.get("compliance_status") == "compliant"
    )
    non_compliant = sum(
        1 for s in suppliers.values() if s.get("compliance_status") == "non_compliant"
    )
    pending = sum(
        1 for s in suppliers.values() if s.get("compliance_status") == "pending"
    )
    compliant_pct = round((compliant / total_suppliers) * 100, 1) if total_suppliers > 0 else 0.0

    # Risk distribution
    risk_dist = RiskDistribution()
    for s in suppliers.values():
        level = s.get("risk_level", "unknown")
        if level == "low":
            risk_dist.low += 1
        elif level == "medium":
            risk_dist.medium += 1
        elif level == "high":
            risk_dist.high += 1
        elif level == "critical":
            risk_dist.critical += 1
        else:
            risk_dist.unknown += 1

    # Plot metrics
    total_plots = len(plots)
    validated = sum(
        1 for p in plots.values() if p.get("validation_status") == "valid"
    )

    # DDS metrics
    total_dds = len(dds_store)
    dds_submitted = sum(
        1 for d in dds_store.values()
        if (d.get("status") == "submitted"
            or (hasattr(d.get("status"), "value") and d["status"].value == "submitted"))
    )
    dds_pending = total_dds - dds_submitted

    # Document metrics
    total_docs = len(documents)
    docs_verified = sum(
        1 for d in documents.values()
        if (d.get("verification_status") == "verified"
            or (hasattr(d.get("verification_status"), "value")
                and d["verification_status"].value == "verified"))
    )

    # Pipeline metrics (today)
    today = datetime.now(timezone.utc).date()
    runs_today = sum(
        1 for r in pipelines.values()
        if r.get("created_at") and r["created_at"].date() == today
    )

    # Alerts
    active_alerts = sum(1 for a in alerts if not a.get("acknowledged", False))

    return DashboardMetrics(
        total_suppliers=total_suppliers,
        compliant_suppliers=compliant,
        compliant_percent=compliant_pct,
        non_compliant_suppliers=non_compliant,
        pending_suppliers=pending,
        total_plots=total_plots,
        validated_plots=validated,
        total_dds=total_dds,
        dds_submitted=dds_submitted,
        dds_pending=dds_pending,
        total_documents=total_docs,
        documents_verified=docs_verified,
        risk_distribution=risk_dist,
        active_alerts=active_alerts,
        pipeline_runs_today=runs_today,
        last_updated=datetime.now(timezone.utc),
    )


@router.get(
    "/trends",
    response_model=ComplianceTrendResponse,
    summary="Compliance trends",
    description="Retrieve monthly compliance trend data for the last 12 months.",
)
async def get_compliance_trends() -> ComplianceTrendResponse:
    """
    Generate monthly compliance trend data.

    In v1.0, data is simulated with a realistic improving trend.
    Production would query time-series aggregates from the database.

    Returns:
        200 with 12-month trend data.
    """
    now = datetime.now(timezone.utc)
    data: List[MonthlyTrend] = []

    random.seed(42)  # Deterministic for consistent demo data
    base_suppliers = 200
    base_compliance = 55.0

    for i in range(12):
        month_date = now - timedelta(days=(11 - i) * 30)
        month_str = month_date.strftime("%Y-%m")

        # Simulate growth and improvement
        suppliers = base_suppliers + i * 8 + random.randint(-3, 5)
        improvement = i * 2.5 + random.uniform(-1.5, 2.0)
        rate = min(95.0, base_compliance + improvement)
        compliant = int(suppliers * rate / 100)

        data.append(MonthlyTrend(
            month=month_str,
            total_suppliers=suppliers,
            compliant_count=compliant,
            compliance_rate=round(rate, 1),
            dds_submitted=compliant + random.randint(0, 10),
            new_alerts=max(0, 15 - i + random.randint(-3, 3)),
            average_risk_score=round(max(20, 65 - i * 2.5 + random.uniform(-3, 3)), 1),
        ))

    # Determine overall trend
    first_half_avg = sum(d.compliance_rate for d in data[:6]) / 6
    second_half_avg = sum(d.compliance_rate for d in data[6:]) / 6

    if second_half_avg > first_half_avg + 3:
        trend = "improving"
    elif second_half_avg < first_half_avg - 3:
        trend = "declining"
    else:
        trend = "stable"

    return ComplianceTrendResponse(
        months=12,
        data=data,
        overall_trend=trend,
    )


@router.get(
    "/alerts",
    response_model=DashboardAlertListResponse,
    summary="Recent alerts",
    description="Retrieve recent alerts and notifications for the dashboard.",
)
async def get_dashboard_alerts() -> DashboardAlertListResponse:
    """
    Retrieve the most recent alerts for display on the dashboard.

    Combines risk alerts with compliance and document alerts. Limited
    to the 20 most recent entries.

    Returns:
        200 with alert list.
    """
    alerts = _get_alert_store()

    # Map to dashboard alert format
    dashboard_alerts: List[DashboardAlert] = []
    category_map = {
        "deforestation": "deforestation",
        "compliance": "compliance",
        "document_expiry": "document",
        "regulatory_change": "regulatory",
    }

    for alert in alerts:
        dashboard_alerts.append(DashboardAlert(
            alert_id=alert["alert_id"],
            severity=alert["severity"],
            title=alert["title"],
            description=alert["description"],
            category=category_map.get(alert.get("alert_type", ""), "compliance"),
            created_at=alert["created_at"],
            acknowledged=alert.get("acknowledged", False),
        ))

    # Sort by recency and take top 20
    dashboard_alerts.sort(key=lambda a: a.created_at, reverse=True)
    dashboard_alerts = dashboard_alerts[:20]

    unacknowledged = sum(1 for a in dashboard_alerts if not a.acknowledged)

    return DashboardAlertListResponse(
        items=dashboard_alerts,
        total=len(dashboard_alerts),
        unacknowledged=unacknowledged,
    )
