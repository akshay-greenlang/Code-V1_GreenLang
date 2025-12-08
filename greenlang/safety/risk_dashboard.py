r"""
RiskDashboard - Real-time Risk Monitoring Dashboard

This module implements a comprehensive risk monitoring dashboard with real-time
metrics, trend analysis, and FastAPI endpoints for process safety management.

Key Features:
- Dashboard class with real-time risk metrics aggregation
- Risk trend analysis with historical tracking
- Heat map visualization data generation
- KPI tracking (open risks, overdue actions, escalations)
- FastAPI endpoints for dashboard data access
- Widget-based dashboard configuration
- Threshold-based alerting

Reference:
- IEC 61511-1:2016 - Functional Safety
- ISO 31000:2018 - Risk Management

Example:
    >>> from greenlang.safety.risk_dashboard import RiskDashboard
    >>> dashboard = RiskDashboard()
    >>> metrics = await dashboard.get_current_metrics()
    >>> print(f"Open Risks: {metrics.total_open_risks}")
"""

from typing import Dict, List, Optional, Any, ClassVar
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import statistics
import asyncio

# FastAPI imports (optional - graceful degradation if not available)
try:
    from fastapi import APIRouter, HTTPException, Query, Depends
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TrendDirection(str, Enum):
    """Direction of metric trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Dashboard alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of dashboard metrics."""

    COUNT = "count"
    PERCENTAGE = "percentage"
    AVERAGE = "average"
    SUM = "sum"
    TREND = "trend"


class WidgetType(str, Enum):
    """Dashboard widget types."""

    KPI = "kpi"  # Key Performance Indicator
    CHART = "chart"  # Line/bar chart
    HEATMAP = "heatmap"  # Risk heatmap
    TABLE = "table"  # Data table
    GAUGE = "gauge"  # Gauge/dial
    LIST = "list"  # Ranked list
    TIMELINE = "timeline"  # Timeline events


# =============================================================================
# DATA MODELS
# =============================================================================

class MetricValue(BaseModel):
    """A single metric value with trend."""

    metric_id: str = Field(..., description="Metric identifier")
    name: str = Field(..., description="Metric display name")
    value: float = Field(..., description="Current value")
    previous_value: Optional[float] = Field(None, description="Previous value")
    change_percent: Optional[float] = Field(None, description="Change percentage")
    trend: TrendDirection = Field(
        default=TrendDirection.UNKNOWN,
        description="Trend direction"
    )
    unit: str = Field(default="", description="Unit of measurement")
    metric_type: MetricType = Field(default=MetricType.COUNT)
    threshold_warning: Optional[float] = Field(None)
    threshold_critical: Optional[float] = Field(None)
    is_warning: bool = Field(default=False)
    is_critical: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DashboardAlert(BaseModel):
    """Dashboard alert notification."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique alert identifier"
    )
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(default="", description="Source of alert")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    related_entity_id: Optional[str] = Field(None)
    related_entity_type: Optional[str] = Field(None)


class HeatmapCell(BaseModel):
    """Single cell in risk heatmap."""

    severity: int = Field(..., ge=1, le=5)
    likelihood: int = Field(..., ge=1, le=5)
    count: int = Field(default=0)
    risk_ids: List[str] = Field(default_factory=list)
    color: str = Field(default="green")


class HeatmapData(BaseModel):
    """Risk heatmap visualization data."""

    cells: List[List[HeatmapCell]] = Field(
        default_factory=list,
        description="5x5 matrix of heatmap cells"
    )
    total_risks: int = Field(default=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TrendDataPoint(BaseModel):
    """Single data point for trend analysis."""

    timestamp: datetime = Field(...)
    value: float = Field(...)
    label: str = Field(default="")


class TrendSeries(BaseModel):
    """Time series data for trend charts."""

    series_id: str = Field(...)
    name: str = Field(...)
    data_points: List[TrendDataPoint] = Field(default_factory=list)
    aggregation: str = Field(default="daily")  # daily, weekly, monthly


class DashboardKPIs(BaseModel):
    """Key Performance Indicators for dashboard."""

    # Risk counts
    total_risks: int = Field(default=0)
    total_open_risks: int = Field(default=0)
    critical_risks: int = Field(default=0)
    high_risks: int = Field(default=0)
    medium_risks: int = Field(default=0)
    low_risks: int = Field(default=0)

    # Action metrics
    total_actions: int = Field(default=0)
    overdue_actions: int = Field(default=0)
    actions_due_this_week: int = Field(default=0)

    # Escalation metrics
    escalated_items: int = Field(default=0)
    pending_verifications: int = Field(default=0)

    # Compliance metrics
    compliance_rate: float = Field(default=100.0)
    on_time_completion_rate: float = Field(default=100.0)

    # Average metrics
    average_risk_score: float = Field(default=0.0)
    average_days_to_close: float = Field(default=0.0)

    # Trend indicators
    risks_added_30d: int = Field(default=0)
    risks_closed_30d: int = Field(default=0)
    risk_trend: TrendDirection = Field(default=TrendDirection.STABLE)

    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_as_of: datetime = Field(default_factory=datetime.utcnow)


class WidgetConfig(BaseModel):
    """Dashboard widget configuration."""

    widget_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique widget identifier"
    )
    widget_type: WidgetType = Field(..., description="Type of widget")
    title: str = Field(..., description="Widget title")
    position: Dict[str, int] = Field(
        default_factory=lambda: {"row": 0, "col": 0, "width": 1, "height": 1},
        description="Grid position"
    )
    data_source: str = Field(default="", description="Data source identifier")
    refresh_interval_seconds: int = Field(default=60)
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Widget-specific configuration"
    )


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    dashboard_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique dashboard identifier"
    )
    name: str = Field(default="Risk Management Dashboard")
    description: str = Field(default="")
    widgets: List[WidgetConfig] = Field(default_factory=list)
    refresh_interval_seconds: int = Field(default=300)
    theme: str = Field(default="light")
    created_by: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DashboardMetrics(BaseModel):
    """Complete dashboard metrics response."""

    kpis: DashboardKPIs = Field(...)
    heatmap: HeatmapData = Field(...)
    alerts: List[DashboardAlert] = Field(default_factory=list)
    trend_data: Dict[str, TrendSeries] = Field(default_factory=dict)
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    status_breakdown: Dict[str, int] = Field(default_factory=dict)
    top_risks: List[Dict[str, Any]] = Field(default_factory=list)
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")


# =============================================================================
# DASHBOARD CLASS
# =============================================================================

class RiskDashboard:
    """
    Real-time Risk Monitoring Dashboard.

    Aggregates data from risk register, action tracker, and safeguard registry
    to provide comprehensive risk monitoring and KPI tracking.

    Attributes:
        config: DashboardConfig
        alerts: List of active alerts
        metric_history: Historical metric values

    Example:
        >>> dashboard = RiskDashboard()
        >>> metrics = await dashboard.get_current_metrics()
        >>> print(f"Critical Risks: {metrics.kpis.critical_risks}")
    """

    # Default threshold configurations
    DEFAULT_THRESHOLDS: ClassVar[Dict[str, Dict[str, float]]] = {
        "critical_risks": {"warning": 3, "critical": 5},
        "overdue_actions": {"warning": 5, "critical": 10},
        "compliance_rate": {"warning": 90, "critical": 80},
        "escalated_items": {"warning": 3, "critical": 5},
    }

    # Risk level colors
    RISK_COLORS: ClassVar[Dict[str, str]] = {
        "low": "#28a745",  # Green
        "medium": "#ffc107",  # Yellow
        "high": "#fd7e14",  # Orange
        "critical": "#dc3545",  # Red
    }

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        risk_register: Optional[Any] = None,
        action_tracker: Optional[Any] = None,
        safeguard_registry: Optional[Any] = None
    ):
        """
        Initialize RiskDashboard.

        Args:
            config: Optional dashboard configuration
            risk_register: Optional RiskRegister instance
            action_tracker: Optional ActionTracker instance
            safeguard_registry: Optional SafeguardRegistry instance
        """
        self.config = config or DashboardConfig()
        self.risk_register = risk_register
        self.action_tracker = action_tracker
        self.safeguard_registry = safeguard_registry

        self.alerts: List[DashboardAlert] = []
        self.metric_history: Dict[str, List[MetricValue]] = defaultdict(list)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60

        logger.info(f"RiskDashboard initialized: {self.config.name}")

    # =========================================================================
    # METRIC CALCULATION
    # =========================================================================

    async def get_current_metrics(
        self,
        use_cache: bool = True
    ) -> DashboardMetrics:
        """
        Get current dashboard metrics.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DashboardMetrics with all current data
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._cache.get("metrics")

        now = datetime.utcnow()

        # Calculate KPIs
        kpis = await self._calculate_kpis()

        # Generate heatmap
        heatmap = await self._generate_heatmap()

        # Get active alerts
        alerts = self._get_active_alerts()

        # Calculate trend data
        trend_data = await self._calculate_trends()

        # Get breakdowns
        category_breakdown = await self._get_category_breakdown()
        status_breakdown = await self._get_status_breakdown()

        # Get top risks
        top_risks = await self._get_top_risks(limit=10)

        # Get recent activity
        recent_activity = await self._get_recent_activity(limit=20)

        # Build response
        metrics = DashboardMetrics(
            kpis=kpis,
            heatmap=heatmap,
            alerts=alerts,
            trend_data=trend_data,
            category_breakdown=category_breakdown,
            status_breakdown=status_breakdown,
            top_risks=top_risks,
            recent_activity=recent_activity,
            generated_at=now,
            provenance_hash=self._calculate_provenance(kpis)
        )

        # Update cache
        self._cache["metrics"] = metrics
        self._cache_timestamp = now

        return metrics

    async def _calculate_kpis(self) -> DashboardKPIs:
        """Calculate Key Performance Indicators."""
        kpis = DashboardKPIs()
        now = datetime.utcnow()
        thirty_days_ago = now - timedelta(days=30)

        # Risk metrics from register
        if self.risk_register:
            risks = list(self.risk_register.risks.values())
            open_risks = [r for r in risks if r.status.value != "closed"]

            kpis.total_risks = len(risks)
            kpis.total_open_risks = len(open_risks)

            # Count by level
            kpis.critical_risks = sum(
                1 for r in open_risks if r.risk_level.value == "critical"
            )
            kpis.high_risks = sum(
                1 for r in open_risks if r.risk_level.value == "high"
            )
            kpis.medium_risks = sum(
                1 for r in open_risks if r.risk_level.value == "medium"
            )
            kpis.low_risks = sum(
                1 for r in open_risks if r.risk_level.value == "low"
            )

            # Average score
            if open_risks:
                kpis.average_risk_score = statistics.mean(
                    [r.risk_score for r in open_risks]
                )

            # 30-day trends
            kpis.risks_added_30d = sum(
                1 for r in risks
                if r.identified_date and r.identified_date > thirty_days_ago
            )
            kpis.risks_closed_30d = sum(
                1 for r in risks
                if r.status.value == "closed" and r.updated_at > thirty_days_ago
            )

            # Calculate trend direction
            if kpis.risks_closed_30d > kpis.risks_added_30d:
                kpis.risk_trend = TrendDirection.IMPROVING
            elif kpis.risks_added_30d > kpis.risks_closed_30d:
                kpis.risk_trend = TrendDirection.WORSENING
            else:
                kpis.risk_trend = TrendDirection.STABLE

        # Action metrics from tracker
        if self.action_tracker:
            actions = list(self.action_tracker.actions.values())

            kpis.total_actions = len(actions)

            # Overdue
            kpis.overdue_actions = sum(
                1 for a in actions
                if a.due_date and a.due_date < now
                and a.status.value not in ["completed", "verified", "cancelled"]
            )

            # Due this week
            week_ahead = now + timedelta(days=7)
            kpis.actions_due_this_week = sum(
                1 for a in actions
                if a.due_date and now <= a.due_date <= week_ahead
                and a.status.value not in ["completed", "verified", "cancelled"]
            )

            # Escalations
            kpis.escalated_items = sum(
                1 for a in actions
                if a.escalation_level.value != "none"
            )

            # On-time completion rate
            completed = [
                a for a in actions
                if a.status.value in ["completed", "verified"]
            ]
            if completed:
                on_time = sum(
                    1 for a in completed
                    if a.completed_date and a.due_date
                    and a.completed_date <= a.due_date
                )
                kpis.on_time_completion_rate = on_time / len(completed) * 100

        # Safeguard metrics
        if self.safeguard_registry:
            verifications = list(self.safeguard_registry.verifications.values())

            kpis.pending_verifications = sum(
                1 for v in verifications
                if v.status.value in ["scheduled", "in_progress", "awaiting_review"]
            )

        # Compliance rate (simplified calculation)
        total_items = kpis.total_open_risks + kpis.total_actions
        non_compliant = kpis.overdue_actions + kpis.escalated_items
        if total_items > 0:
            kpis.compliance_rate = (1 - non_compliant / total_items) * 100
            kpis.compliance_rate = max(0, min(100, kpis.compliance_rate))

        kpis.last_updated = now
        kpis.data_as_of = now

        return kpis

    async def _generate_heatmap(self) -> HeatmapData:
        """Generate risk heatmap data."""
        # Initialize 5x5 matrix
        cells = []
        for severity in range(1, 6):
            row = []
            for likelihood in range(1, 6):
                cell = HeatmapCell(
                    severity=severity,
                    likelihood=likelihood,
                    count=0,
                    risk_ids=[],
                    color=self._get_cell_color(severity, likelihood)
                )
                row.append(cell)
            cells.append(row)

        total_risks = 0

        if self.risk_register:
            for risk in self.risk_register.risks.values():
                if risk.status.value == "closed":
                    continue

                s_idx = risk.severity - 1
                l_idx = risk.likelihood - 1
                cells[s_idx][l_idx].count += 1
                cells[s_idx][l_idx].risk_ids.append(risk.risk_id)
                total_risks += 1

        return HeatmapData(
            cells=cells,
            total_risks=total_risks,
            generated_at=datetime.utcnow()
        )

    def _get_cell_color(self, severity: int, likelihood: int) -> str:
        """Get heatmap cell color based on risk level."""
        # Risk matrix mapping
        score = severity * likelihood

        if score >= 15:
            return self.RISK_COLORS["critical"]
        elif score >= 9:
            return self.RISK_COLORS["high"]
        elif score >= 4:
            return self.RISK_COLORS["medium"]
        else:
            return self.RISK_COLORS["low"]

    async def _calculate_trends(self) -> Dict[str, TrendSeries]:
        """Calculate trend data for charts."""
        trends = {}
        now = datetime.utcnow()

        # Risk count trend (last 30 days)
        risk_trend = TrendSeries(
            series_id="risk_count",
            name="Open Risks",
            aggregation="daily"
        )

        if self.risk_register:
            # Simulate daily counts (in production, this would come from historical data)
            for i in range(30, -1, -1):
                date = now - timedelta(days=i)
                # Simplified: count risks that existed on that date
                count = sum(
                    1 for r in self.risk_register.risks.values()
                    if r.identified_date and r.identified_date <= date
                    and (r.status.value != "closed" or
                         (hasattr(r, 'closed_date') and r.closed_date and r.closed_date > date))
                )
                risk_trend.data_points.append(TrendDataPoint(
                    timestamp=date,
                    value=count,
                    label=date.strftime("%Y-%m-%d")
                ))

        trends["risk_count"] = risk_trend

        # Action completion trend
        action_trend = TrendSeries(
            series_id="actions_completed",
            name="Actions Completed",
            aggregation="weekly"
        )

        if self.action_tracker:
            for week in range(4, -1, -1):
                week_start = now - timedelta(weeks=week + 1)
                week_end = now - timedelta(weeks=week)
                count = sum(
                    1 for a in self.action_tracker.actions.values()
                    if a.completed_date and week_start <= a.completed_date < week_end
                )
                action_trend.data_points.append(TrendDataPoint(
                    timestamp=week_end,
                    value=count,
                    label=f"Week {4 - week}"
                ))

        trends["actions_completed"] = action_trend

        return trends

    async def _get_category_breakdown(self) -> Dict[str, int]:
        """Get risk count by category."""
        breakdown = defaultdict(int)

        if self.risk_register:
            for risk in self.risk_register.risks.values():
                if risk.status.value != "closed":
                    breakdown[risk.category.value] += 1

        return dict(breakdown)

    async def _get_status_breakdown(self) -> Dict[str, int]:
        """Get risk count by status."""
        breakdown = defaultdict(int)

        if self.risk_register:
            for risk in self.risk_register.risks.values():
                breakdown[risk.status.value] += 1

        return dict(breakdown)

    async def _get_top_risks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top risks by score."""
        top_risks = []

        if self.risk_register:
            open_risks = [
                r for r in self.risk_register.risks.values()
                if r.status.value != "closed"
            ]
            sorted_risks = sorted(
                open_risks,
                key=lambda x: x.risk_score,
                reverse=True
            )[:limit]

            for risk in sorted_risks:
                top_risks.append({
                    "risk_id": risk.risk_id,
                    "title": risk.title,
                    "severity": risk.severity,
                    "likelihood": risk.likelihood,
                    "risk_level": risk.risk_level.value,
                    "risk_score": risk.risk_score,
                    "category": risk.category.value,
                    "status": risk.status.value,
                    "target_date": risk.target_mitigation_date.isoformat()
                    if risk.target_mitigation_date else None
                })

        return top_risks

    async def _get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity across all systems."""
        activity = []
        now = datetime.utcnow()

        # Risk activity
        if self.risk_register:
            for event in self.risk_register.audit_trail[-limit:]:
                activity.append({
                    "type": "risk",
                    "event": event.get("event_type", ""),
                    "entity_id": event.get("risk_id", ""),
                    "timestamp": event.get("timestamp", now).isoformat(),
                    "details": event.get("details", {})
                })

        # Action activity
        if self.action_tracker:
            for event in self.action_tracker.audit_trail[-limit:]:
                activity.append({
                    "type": "action",
                    "event": event.get("event_type", ""),
                    "entity_id": event.get("action_id", ""),
                    "timestamp": event.get("timestamp", now).isoformat(),
                    "details": event.get("details", {})
                })

        # Sort by timestamp and limit
        activity.sort(key=lambda x: x["timestamp"], reverse=True)
        return activity[:limit]

    # =========================================================================
    # ALERTS
    # =========================================================================

    def _get_active_alerts(self) -> List[DashboardAlert]:
        """Get current active alerts based on thresholds."""
        alerts = []

        if self.risk_register:
            # Critical risks alert
            critical_count = sum(
                1 for r in self.risk_register.risks.values()
                if r.risk_level.value == "critical" and r.status.value != "closed"
            )
            thresholds = self.DEFAULT_THRESHOLDS["critical_risks"]

            if critical_count >= thresholds["critical"]:
                alerts.append(DashboardAlert(
                    severity=AlertSeverity.CRITICAL,
                    title="High Number of Critical Risks",
                    message=f"{critical_count} critical risks require immediate attention",
                    source="risk_register"
                ))
            elif critical_count >= thresholds["warning"]:
                alerts.append(DashboardAlert(
                    severity=AlertSeverity.WARNING,
                    title="Critical Risks Elevated",
                    message=f"{critical_count} critical risks in the system",
                    source="risk_register"
                ))

        if self.action_tracker:
            # Overdue actions alert
            now = datetime.utcnow()
            overdue_count = sum(
                1 for a in self.action_tracker.actions.values()
                if a.due_date and a.due_date < now
                and a.status.value not in ["completed", "verified", "cancelled"]
            )
            thresholds = self.DEFAULT_THRESHOLDS["overdue_actions"]

            if overdue_count >= thresholds["critical"]:
                alerts.append(DashboardAlert(
                    severity=AlertSeverity.ERROR,
                    title="Many Actions Overdue",
                    message=f"{overdue_count} actions are past their due date",
                    source="action_tracker"
                ))
            elif overdue_count >= thresholds["warning"]:
                alerts.append(DashboardAlert(
                    severity=AlertSeverity.WARNING,
                    title="Actions Approaching Deadline",
                    message=f"{overdue_count} actions are overdue",
                    source="action_tracker"
                ))

        # Include stored alerts
        alerts.extend([a for a in self.alerts if not a.acknowledged])

        return alerts

    def add_alert(self, alert: DashboardAlert) -> DashboardAlert:
        """Add a new alert."""
        self.alerts.append(alert)
        logger.info(f"Alert added: {alert.title} ({alert.severity.value})")
        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> Optional[DashboardAlert]:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                logger.info(f"Alert acknowledged: {alert_id}")
                return alert
        return None

    # =========================================================================
    # WIDGET DATA
    # =========================================================================

    async def get_widget_data(
        self,
        widget_id: str
    ) -> Dict[str, Any]:
        """
        Get data for a specific widget.

        Args:
            widget_id: Widget identifier

        Returns:
            Widget-specific data dictionary
        """
        # Find widget config
        widget = None
        for w in self.config.widgets:
            if w.widget_id == widget_id:
                widget = w
                break

        if not widget:
            return {"error": "Widget not found"}

        data = {
            "widget_id": widget_id,
            "widget_type": widget.widget_type.value,
            "title": widget.title,
            "generated_at": datetime.utcnow().isoformat()
        }

        # Generate data based on widget type
        if widget.widget_type == WidgetType.KPI:
            kpis = await self._calculate_kpis()
            data["kpis"] = kpis.model_dump()

        elif widget.widget_type == WidgetType.HEATMAP:
            heatmap = await self._generate_heatmap()
            data["heatmap"] = heatmap.model_dump()

        elif widget.widget_type == WidgetType.CHART:
            trends = await self._calculate_trends()
            series_id = widget.config.get("series_id", "risk_count")
            if series_id in trends:
                data["series"] = trends[series_id].model_dump()

        elif widget.widget_type == WidgetType.TABLE:
            limit = widget.config.get("limit", 10)
            data["rows"] = await self._get_top_risks(limit)

        elif widget.widget_type == WidgetType.LIST:
            limit = widget.config.get("limit", 5)
            data["items"] = await self._get_top_risks(limit)

        elif widget.widget_type == WidgetType.TIMELINE:
            limit = widget.config.get("limit", 10)
            data["events"] = await self._get_recent_activity(limit)

        return data

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp:
            return False
        age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self._cache = {}
        self._cache_timestamp = None
        logger.debug("Dashboard cache cleared")

    def set_cache_ttl(self, seconds: int) -> None:
        """Set cache time-to-live in seconds."""
        self._cache_ttl_seconds = seconds

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_provenance(self, kpis: DashboardKPIs) -> str:
        """Calculate provenance hash for dashboard data."""
        data_str = (
            f"{kpis.total_risks}|"
            f"{kpis.critical_risks}|"
            f"{kpis.overdue_actions}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_config(self) -> DashboardConfig:
        """Get dashboard configuration."""
        return self.config

    def update_config(self, config: DashboardConfig) -> DashboardConfig:
        """Update dashboard configuration."""
        self.config = config
        self.config.updated_at = datetime.utcnow()
        self.clear_cache()
        return self.config


# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

def create_dashboard_router(dashboard: RiskDashboard) -> Optional[Any]:
    """
    Create FastAPI router for dashboard endpoints.

    Args:
        dashboard: RiskDashboard instance

    Returns:
        FastAPI APIRouter or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - dashboard endpoints disabled")
        return None

    router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

    @router.get("/metrics", response_model=DashboardMetrics)
    async def get_metrics(
        use_cache: bool = Query(True, description="Use cached data if available")
    ):
        """Get current dashboard metrics."""
        try:
            return await dashboard.get_current_metrics(use_cache=use_cache)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/kpis", response_model=DashboardKPIs)
    async def get_kpis():
        """Get Key Performance Indicators only."""
        try:
            return await dashboard._calculate_kpis()
        except Exception as e:
            logger.error(f"Error getting KPIs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/heatmap", response_model=HeatmapData)
    async def get_heatmap():
        """Get risk heatmap data."""
        try:
            return await dashboard._generate_heatmap()
        except Exception as e:
            logger.error(f"Error getting heatmap: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/alerts")
    async def get_alerts(include_acknowledged: bool = False):
        """Get active alerts."""
        try:
            alerts = dashboard._get_active_alerts()
            if not include_acknowledged:
                alerts = [a for a in alerts if not a.acknowledged]
            return {"alerts": [a.model_dump() for a in alerts]}
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(
        alert_id: str,
        acknowledged_by: str = Query(..., description="Person acknowledging")
    ):
        """Acknowledge an alert."""
        try:
            alert = dashboard.acknowledge_alert(alert_id, acknowledged_by)
            if alert:
                return {"status": "acknowledged", "alert": alert.model_dump()}
            raise HTTPException(status_code=404, detail="Alert not found")
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trends")
    async def get_trends():
        """Get trend data for charts."""
        try:
            trends = await dashboard._calculate_trends()
            return {
                "trends": {k: v.model_dump() for k, v in trends.items()}
            }
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/top-risks")
    async def get_top_risks(limit: int = Query(10, ge=1, le=50)):
        """Get top risks by score."""
        try:
            risks = await dashboard._get_top_risks(limit)
            return {"risks": risks, "count": len(risks)}
        except Exception as e:
            logger.error(f"Error getting top risks: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/activity")
    async def get_activity(limit: int = Query(20, ge=1, le=100)):
        """Get recent activity."""
        try:
            activity = await dashboard._get_recent_activity(limit)
            return {"activity": activity, "count": len(activity)}
        except Exception as e:
            logger.error(f"Error getting activity: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/widget/{widget_id}")
    async def get_widget_data(widget_id: str):
        """Get data for a specific widget."""
        try:
            return await dashboard.get_widget_data(widget_id)
        except Exception as e:
            logger.error(f"Error getting widget data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/cache/clear")
    async def clear_cache():
        """Clear the dashboard cache."""
        dashboard.clear_cache()
        return {"status": "cache_cleared"}

    @router.get("/config")
    async def get_config():
        """Get dashboard configuration."""
        return dashboard.get_config().model_dump()

    return router


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_dashboard_config() -> DashboardConfig:
    """Create a default dashboard configuration with standard widgets."""
    widgets = [
        WidgetConfig(
            widget_type=WidgetType.KPI,
            title="Key Performance Indicators",
            position={"row": 0, "col": 0, "width": 4, "height": 1}
        ),
        WidgetConfig(
            widget_type=WidgetType.HEATMAP,
            title="Risk Heatmap",
            position={"row": 1, "col": 0, "width": 2, "height": 2}
        ),
        WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Risk Trend",
            position={"row": 1, "col": 2, "width": 2, "height": 2},
            config={"series_id": "risk_count"}
        ),
        WidgetConfig(
            widget_type=WidgetType.TABLE,
            title="Top Risks",
            position={"row": 3, "col": 0, "width": 2, "height": 2},
            config={"limit": 10}
        ),
        WidgetConfig(
            widget_type=WidgetType.TIMELINE,
            title="Recent Activity",
            position={"row": 3, "col": 2, "width": 2, "height": 2},
            config={"limit": 10}
        ),
    ]

    return DashboardConfig(
        name="Risk Management Dashboard",
        description="Real-time risk monitoring and KPI tracking",
        widgets=widgets,
        refresh_interval_seconds=300
    )


if __name__ == "__main__":
    # Example usage
    print("RiskDashboard module loaded successfully")

    # Create dashboard with default config
    config = create_default_dashboard_config()
    dashboard = RiskDashboard(config=config)

    # Note: In production, run with actual data sources
    print(f"Dashboard: {dashboard.config.name}")
    print(f"Widgets: {len(dashboard.config.widgets)}")
