# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Compliance Dashboard Template
==========================================================

Generates a real-time EUDR compliance KPI dashboard covering overall
compliance scores, commodity breakdowns, risk distributions, supplier
status, geolocation coverage, certification summaries, data quality
metrics, upcoming deadlines, and recent activity.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "compliance_dashboard"
TEMPLATE_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class CommodityType(str, Enum):
    """EUDR-regulated commodity types."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class ComplianceTrafficLight(str, Enum):
    """Traffic light compliance status."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class RiskLevel(str, Enum):
    """Supplier risk level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DDSSubmissionStatus(str, Enum):
    """DDS submission status."""
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    OVERDUE = "OVERDUE"
    DRAFT = "DRAFT"


class EventType(str, Enum):
    """Compliance event types."""
    DDS_SUBMITTED = "DDS_SUBMITTED"
    SUPPLIER_ADDED = "SUPPLIER_ADDED"
    RISK_CHANGED = "RISK_CHANGED"
    CERT_RENEWED = "CERT_RENEWED"
    GEO_VALIDATED = "GEO_VALIDATED"
    DATA_UPDATED = "DATA_UPDATED"
    ALERT_RAISED = "ALERT_RAISED"
    DD_COMPLETED = "DD_COMPLETED"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ComplianceOverview(BaseModel):
    """Section 1: Overall compliance metrics."""
    overall_compliance_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall EUDR compliance score"
    )
    dds_submitted_count: int = Field(0, ge=0, description="DDS submitted")
    dds_pending_count: int = Field(0, ge=0, description="DDS pending")
    dds_overdue_count: int = Field(0, ge=0, description="DDS overdue")
    dds_draft_count: int = Field(0, ge=0, description="DDS in draft")


class CommodityComplianceEntry(BaseModel):
    """Compliance status for a single commodity."""
    commodity: CommodityType = Field(..., description="Commodity type")
    status: ComplianceTrafficLight = Field(..., description="Traffic light status")
    suppliers_count: int = Field(0, ge=0, description="Number of suppliers")
    dds_count: int = Field(0, ge=0, description="DDS count for this commodity")
    compliance_pct: float = Field(0.0, ge=0.0, le=100.0, description="Compliance %")
    volume_kg: Optional[float] = Field(None, ge=0, description="Volume in kg")


class RiskDistributionEntry(BaseModel):
    """Risk distribution data point."""
    risk_level: RiskLevel = Field(..., description="Risk level")
    supplier_count: int = Field(0, ge=0, description="Suppliers at this level")
    percentage: float = Field(0.0, ge=0.0, le=100.0, description="Percentage")


class SupplierStatusSummary(BaseModel):
    """Section 4: Supplier status metrics."""
    total_suppliers: int = Field(0, ge=0, description="Total registered suppliers")
    onboarded: int = Field(0, ge=0, description="Fully onboarded")
    dd_complete: int = Field(0, ge=0, description="Due diligence complete")
    dd_pending: int = Field(0, ge=0, description="Due diligence pending")
    dd_overdue: int = Field(0, ge=0, description="Due diligence overdue")


class GeolocationCoverage(BaseModel):
    """Section 5: Geolocation coverage metrics."""
    plots_validated: int = Field(0, ge=0, description="Plots validated")
    plots_pending: int = Field(0, ge=0, description="Plots pending validation")
    plots_failed: int = Field(0, ge=0, description="Plots failed validation")
    total_area_ha: float = Field(0.0, ge=0.0, description="Total area in hectares")
    countries_covered: int = Field(0, ge=0, description="Countries with plots")
    country_list: List[str] = Field(
        default_factory=list, description="ISO codes of covered countries"
    )


class CertificationEntry(BaseModel):
    """Active certification by scheme."""
    scheme: str = Field(..., description="Certification scheme name")
    active_count: int = Field(0, ge=0, description="Active certificates")
    expired_count: int = Field(0, ge=0, description="Expired certificates")
    expiring_soon_count: int = Field(0, ge=0, description="Expiring within 90 days")


class DataQualityMetric(BaseModel):
    """Data quality metric by category."""
    category: str = Field(..., description="Data quality category")
    score: float = Field(0.0, ge=0.0, le=100.0, description="Quality score 0-100")
    records_total: int = Field(0, ge=0, description="Total records")
    records_valid: int = Field(0, ge=0, description="Valid records")
    issues_count: int = Field(0, ge=0, description="Quality issues found")


class UpcomingDeadline(BaseModel):
    """Upcoming compliance deadline."""
    deadline_id: str = Field(..., description="Deadline identifier")
    title: str = Field(..., description="Deadline title")
    deadline_date: date = Field(..., description="Due date")
    category: str = Field("", description="Category (DDS/Certification/Review)")
    days_remaining: int = Field(0, description="Days until deadline")
    readiness_pct: float = Field(0.0, ge=0.0, le=100.0, description="Readiness")
    owner: Optional[str] = Field(None, description="Responsible party")


class RecentActivity(BaseModel):
    """Recent compliance activity event."""
    event_id: str = Field(..., description="Event identifier")
    event_type: EventType = Field(..., description="Event type")
    description: str = Field(..., description="Event description")
    timestamp: datetime = Field(..., description="Event timestamp")
    actor: Optional[str] = Field(None, description="Who performed the action")
    entity_reference: Optional[str] = Field(None, description="Related entity ID")


class ComplianceDashboardInput(BaseModel):
    """Complete input data for the EUDR Compliance Dashboard."""
    company_name: str = Field(..., description="Reporting entity")
    dashboard_date: date = Field(
        default_factory=date.today, description="Dashboard snapshot date"
    )
    overview: ComplianceOverview = Field(
        default_factory=ComplianceOverview, description="Overall compliance"
    )
    commodity_breakdown: List[CommodityComplianceEntry] = Field(
        default_factory=list, description="Per-commodity compliance"
    )
    risk_distribution: List[RiskDistributionEntry] = Field(
        default_factory=list, description="Risk distribution data"
    )
    supplier_status: SupplierStatusSummary = Field(
        default_factory=SupplierStatusSummary, description="Supplier status"
    )
    geolocation_coverage: GeolocationCoverage = Field(
        default_factory=GeolocationCoverage, description="Geolocation coverage"
    )
    certifications: List[CertificationEntry] = Field(
        default_factory=list, description="Certification summary"
    )
    data_quality: List[DataQualityMetric] = Field(
        default_factory=list, description="Data quality metrics"
    )
    upcoming_deadlines: List[UpcomingDeadline] = Field(
        default_factory=list, description="Upcoming deadlines"
    )
    recent_activity: List[RecentActivity] = Field(
        default_factory=list, description="Recent activity (max 10)"
    )

    @field_validator("recent_activity")
    @classmethod
    def limit_recent_activity(cls, v: List[RecentActivity]) -> List[RecentActivity]:
        """Limit recent activity to 10 most recent events."""
        return sorted(v, key=lambda e: e.timestamp, reverse=True)[:10]

    @property
    def overall_data_quality_score(self) -> float:
        """Calculate overall data quality score."""
        if not self.data_quality:
            return 0.0
        return sum(m.score for m in self.data_quality) / len(self.data_quality)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _traffic_light_badge(status: ComplianceTrafficLight) -> str:
    """Text badge for traffic light status."""
    return f"[{status.value}]"


def _traffic_light_css(status: ComplianceTrafficLight) -> str:
    """CSS class for traffic light status."""
    mapping = {
        ComplianceTrafficLight.GREEN: "color:#1a7f37;font-weight:bold;",
        ComplianceTrafficLight.AMBER: "color:#b08800;font-weight:bold;",
        ComplianceTrafficLight.RED: "color:#cf222e;font-weight:bold;",
    }
    return mapping.get(status, "")


def _risk_badge(level: RiskLevel) -> str:
    """Text badge for risk level."""
    return f"[{level.value}]"


def _risk_css(level: RiskLevel) -> str:
    """CSS class for risk level."""
    mapping = {
        RiskLevel.LOW: "color:#1a7f37;",
        RiskLevel.MEDIUM: "color:#b08800;",
        RiskLevel.HIGH: "color:#e36209;",
        RiskLevel.CRITICAL: "color:#cf222e;font-weight:bold;",
    }
    return mapping.get(level, "")


def _fmt_commodity(commodity: CommodityType) -> str:
    """Human-readable commodity name."""
    mapping = {
        CommodityType.CATTLE: "Cattle",
        CommodityType.COCOA: "Cocoa",
        CommodityType.COFFEE: "Coffee",
        CommodityType.OIL_PALM: "Oil Palm",
        CommodityType.RUBBER: "Rubber",
        CommodityType.SOYA: "Soya",
        CommodityType.WOOD: "Wood",
    }
    return mapping.get(commodity, commodity.value)


def _fmt_volume(kg: Optional[float]) -> str:
    """Format volume with appropriate scale."""
    if kg is None:
        return "N/A"
    if kg >= 1_000_000:
        return f"{kg / 1_000:,.0f} t"
    if kg >= 1_000:
        return f"{kg / 1_000:,.1f} t"
    return f"{kg:,.0f} kg"


def _fmt_area(ha: float) -> str:
    """Format area in hectares."""
    if ha >= 1_000_000:
        return f"{ha / 1_000_000:,.1f}M ha"
    if ha >= 1_000:
        return f"{ha / 1_000:,.1f}K ha"
    return f"{ha:,.1f} ha"


def _deadline_urgency(days: int) -> str:
    """Urgency label for deadline."""
    if days <= 0:
        return "OVERDUE"
    if days <= 7:
        return "URGENT"
    if days <= 30:
        return "SOON"
    return "OK"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ComplianceDashboard:
    """Generate EUDR compliance KPI dashboard.

    Widgets:
        1. Compliance Overview - Overall score, DDS counts
        2. Commodity Breakdown - Per-commodity traffic lights
        3. Risk Distribution - Supplier risk pie chart data
        4. Supplier Status - Onboarding and DD metrics
        5. Geolocation Coverage - Plot validation metrics
        6. Certification Summary - Active certs by scheme
        7. Data Quality Score - Per-category quality metrics
        8. Upcoming Deadlines - Next submissions and renewals
        9. Recent Activity - Last 10 compliance events

    Example:
        >>> dashboard = ComplianceDashboard()
        >>> data = ComplianceDashboardInput(company_name="Acme", ...)
        >>> md = dashboard.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the compliance dashboard template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: ComplianceDashboardInput) -> str:
        """Render the dashboard as Markdown.

        Args:
            data: Validated dashboard input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_compliance_overview(data),
            self._md_commodity_breakdown(data),
            self._md_risk_distribution(data),
            self._md_supplier_status(data),
            self._md_geolocation_coverage(data),
            self._md_certification_summary(data),
            self._md_data_quality(data),
            self._md_upcoming_deadlines(data),
            self._md_recent_activity(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ComplianceDashboardInput) -> str:
        """Render the dashboard as HTML.

        Args:
            data: Validated dashboard input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_compliance_overview(data),
            self._html_commodity_breakdown(data),
            self._html_risk_distribution(data),
            self._html_supplier_status(data),
            self._html_geolocation_coverage(data),
            self._html_certification_summary(data),
            self._html_data_quality(data),
            self._html_upcoming_deadlines(data),
            self._html_recent_activity(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: ComplianceDashboardInput) -> Dict[str, Any]:
        """Render the dashboard as a JSON-serializable dictionary.

        Args:
            data: Validated dashboard input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        return {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "company_name": data.company_name,
            "dashboard_date": data.dashboard_date.isoformat(),
            "widgets": {
                "compliance_overview": data.overview.model_dump(mode="json"),
                "commodity_breakdown": [
                    c.model_dump(mode="json") for c in data.commodity_breakdown
                ],
                "risk_distribution": [
                    r.model_dump(mode="json") for r in data.risk_distribution
                ],
                "supplier_status": data.supplier_status.model_dump(mode="json"),
                "geolocation_coverage": data.geolocation_coverage.model_dump(
                    mode="json"
                ),
                "certifications": [
                    c.model_dump(mode="json") for c in data.certifications
                ],
                "data_quality": {
                    "overall_score": data.overall_data_quality_score,
                    "metrics": [
                        m.model_dump(mode="json") for m in data.data_quality
                    ],
                },
                "upcoming_deadlines": [
                    d.model_dump(mode="json") for d in data.upcoming_deadlines
                ],
                "recent_activity": [
                    {
                        **a.model_dump(mode="json"),
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in data.recent_activity
                ],
            },
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: ComplianceDashboardInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ComplianceDashboardInput) -> str:
        """Dashboard header."""
        return (
            f"# EUDR Compliance Dashboard - {data.company_name}\n"
            f"**Snapshot Date:** {data.dashboard_date.isoformat()}\n\n---"
        )

    def _md_compliance_overview(self, data: ComplianceDashboardInput) -> str:
        """Section 1: Compliance Overview."""
        ov = data.overview
        total_dds = (
            ov.dds_submitted_count + ov.dds_pending_count
            + ov.dds_overdue_count + ov.dds_draft_count
        )
        return (
            "## 1. Compliance Overview\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Overall Compliance Score | **{ov.overall_compliance_score:.1f}%** |\n"
            f"| Total DDS | {total_dds} |\n"
            f"| DDS Submitted | {ov.dds_submitted_count} |\n"
            f"| DDS Pending | {ov.dds_pending_count} |\n"
            f"| DDS Overdue | {ov.dds_overdue_count} |\n"
            f"| DDS Draft | {ov.dds_draft_count} |"
        )

    def _md_commodity_breakdown(self, data: ComplianceDashboardInput) -> str:
        """Section 2: Commodity Breakdown."""
        lines = [
            "## 2. Commodity Breakdown\n",
            "| Commodity | Status | Suppliers | DDS | Compliance | Volume |",
            "|-----------|--------|-----------|-----|------------|--------|",
        ]
        for c in data.commodity_breakdown:
            volume = _fmt_volume(c.volume_kg)
            lines.append(
                f"| {_fmt_commodity(c.commodity)} | {_traffic_light_badge(c.status)} "
                f"| {c.suppliers_count} | {c.dds_count} "
                f"| {c.compliance_pct:.0f}% | {volume} |"
            )
        if not data.commodity_breakdown:
            lines.append("| - | No commodities tracked | - | - | - | - |")
        return "\n".join(lines)

    def _md_risk_distribution(self, data: ComplianceDashboardInput) -> str:
        """Section 3: Risk Distribution."""
        lines = [
            "## 3. Risk Distribution\n",
            "| Risk Level | Suppliers | Percentage |",
            "|------------|-----------|------------|",
        ]
        for r in data.risk_distribution:
            bar = "#" * int(r.percentage / 5) if r.percentage > 0 else "-"
            lines.append(
                f"| {_risk_badge(r.risk_level)} | {r.supplier_count} "
                f"| {r.percentage:.1f}% {bar} |"
            )
        if not data.risk_distribution:
            lines.append("| - | No risk data | - |")
        return "\n".join(lines)

    def _md_supplier_status(self, data: ComplianceDashboardInput) -> str:
        """Section 4: Supplier Status."""
        ss = data.supplier_status
        return (
            "## 4. Supplier Status\n\n"
            "| Metric | Count |\n"
            "|--------|-------|\n"
            f"| Total Suppliers | {ss.total_suppliers} |\n"
            f"| Onboarded | {ss.onboarded} |\n"
            f"| DD Complete | {ss.dd_complete} |\n"
            f"| DD Pending | {ss.dd_pending} |\n"
            f"| DD Overdue | {ss.dd_overdue} |"
        )

    def _md_geolocation_coverage(self, data: ComplianceDashboardInput) -> str:
        """Section 5: Geolocation Coverage."""
        gc = data.geolocation_coverage
        countries = ", ".join(gc.country_list) if gc.country_list else "N/A"
        total_plots = gc.plots_validated + gc.plots_pending + gc.plots_failed
        return (
            "## 5. Geolocation Coverage\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Total Plots | {total_plots} |\n"
            f"| Validated | {gc.plots_validated} |\n"
            f"| Pending | {gc.plots_pending} |\n"
            f"| Failed | {gc.plots_failed} |\n"
            f"| Total Area | {_fmt_area(gc.total_area_ha)} |\n"
            f"| Countries Covered | {gc.countries_covered} |\n"
            f"| Country List | {countries} |"
        )

    def _md_certification_summary(self, data: ComplianceDashboardInput) -> str:
        """Section 6: Certification Summary."""
        lines = [
            "## 6. Certification Summary\n",
            "| Scheme | Active | Expired | Expiring Soon |",
            "|--------|--------|---------|---------------|",
        ]
        for c in data.certifications:
            lines.append(
                f"| {c.scheme} | {c.active_count} | {c.expired_count} "
                f"| {c.expiring_soon_count} |"
            )
        if not data.certifications:
            lines.append("| - | No certifications tracked | - | - |")
        return "\n".join(lines)

    def _md_data_quality(self, data: ComplianceDashboardInput) -> str:
        """Section 7: Data Quality Score."""
        overall = data.overall_data_quality_score
        lines = [
            "## 7. Data Quality Score\n",
            f"**Overall Data Quality:** {overall:.1f}%\n",
            "| Category | Score | Records (Valid/Total) | Issues |",
            "|----------|-------|----------------------|--------|",
        ]
        for m in data.data_quality:
            lines.append(
                f"| {m.category} | {m.score:.1f}% "
                f"| {m.records_valid}/{m.records_total} | {m.issues_count} |"
            )
        if not data.data_quality:
            lines.append("| - | No data quality metrics | - | - |")
        return "\n".join(lines)

    def _md_upcoming_deadlines(self, data: ComplianceDashboardInput) -> str:
        """Section 8: Upcoming Deadlines."""
        lines = [
            "## 8. Upcoming Deadlines\n",
            "| ID | Title | Category | Deadline | Days Left | Readiness | Owner |",
            "|----|-------|----------|----------|-----------|-----------|-------|",
        ]
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            owner = d.owner or "TBD"
            urgency = _deadline_urgency(d.days_remaining)
            lines.append(
                f"| {d.deadline_id} | {d.title} | {d.category} "
                f"| {d.deadline_date.isoformat()} | {d.days_remaining} [{urgency}] "
                f"| {d.readiness_pct:.0f}% | {owner} |"
            )
        if not data.upcoming_deadlines:
            lines.append("| - | No upcoming deadlines | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_recent_activity(self, data: ComplianceDashboardInput) -> str:
        """Section 9: Recent Activity."""
        lines = [
            "## 9. Recent Activity\n",
            "| Event | Type | Description | Time | Actor |",
            "|-------|------|-------------|------|-------|",
        ]
        for a in data.recent_activity:
            actor = a.actor or "System"
            ts = a.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"| {a.event_id} | {a.event_type.value} "
                f"| {a.description} | {ts} | {actor} |"
            )
        if not data.recent_activity:
            lines.append("| - | No recent activity | - | - | - |")
        return "\n".join(lines)

    def _md_provenance(self, data: ComplianceDashboardInput) -> str:
        """Provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: ComplianceDashboardInput, body: str) -> str:
        """Wrap body in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>EUDR Dashboard - {data.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1200px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".kpi-card{display:inline-block;text-align:center;padding:1rem 2rem;"
            "border:2px solid #e2e8f0;border-radius:8px;margin:0.5rem;"
            "min-width:150px;}\n"
            ".kpi-value{font-size:2rem;font-weight:bold;}\n"
            ".kpi-label{font-size:0.85rem;color:#666;}\n"
            ".traffic-green{color:#1a7f37;font-weight:bold;}\n"
            ".traffic-amber{color:#b08800;font-weight:bold;}\n"
            ".traffic-red{color:#cf222e;font-weight:bold;}\n"
            ".risk-low{color:#1a7f37;}\n"
            ".risk-medium{color:#b08800;}\n"
            ".risk-high{color:#e36209;}\n"
            ".risk-critical{color:#cf222e;font-weight:bold;}\n"
            ".bar{height:16px;border-radius:3px;display:inline-block;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: ComplianceDashboardInput) -> str:
        """HTML dashboard header."""
        return (
            '<div class="section">\n'
            f"<h1>EUDR Compliance Dashboard &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Snapshot Date:</strong> "
            f"{data.dashboard_date.isoformat()}</p>\n<hr>\n</div>"
        )

    def _html_compliance_overview(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 1: Compliance Overview KPI cards."""
        ov = data.overview
        total_dds = (
            ov.dds_submitted_count + ov.dds_pending_count
            + ov.dds_overdue_count + ov.dds_draft_count
        )
        return (
            '<div class="section">\n<h2>1. Compliance Overview</h2>\n<div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f'{ov.overall_compliance_score:.0f}%</div>'
            f'<div class="kpi-label">Compliance Score</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">{total_dds}</div>'
            f'<div class="kpi-label">Total DDS</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f'{ov.dds_submitted_count}</div>'
            f'<div class="kpi-label">Submitted</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f'{ov.dds_pending_count}</div>'
            f'<div class="kpi-label">Pending</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f'{ov.dds_overdue_count}</div>'
            f'<div class="kpi-label">Overdue</div></div>\n'
            "</div>\n</div>"
        )

    def _html_commodity_breakdown(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 2: Commodity Breakdown."""
        rows = ""
        for c in data.commodity_breakdown:
            css = _traffic_light_css(c.status)
            volume = _fmt_volume(c.volume_kg)
            rows += (
                f"<tr><td>{_fmt_commodity(c.commodity)}</td>"
                f'<td style="{css}">{c.status.value}</td>'
                f"<td>{c.suppliers_count}</td><td>{c.dds_count}</td>"
                f"<td>{c.compliance_pct:.0f}%</td><td>{volume}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No commodities tracked</td></tr>'
        return (
            '<div class="section">\n<h2>2. Commodity Breakdown</h2>\n'
            "<table><thead><tr><th>Commodity</th><th>Status</th>"
            "<th>Suppliers</th><th>DDS</th><th>Compliance</th>"
            f"<th>Volume</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_risk_distribution(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 3: Risk Distribution."""
        rows = ""
        for r in data.risk_distribution:
            css = _risk_css(r.risk_level)
            bar_width = max(r.percentage, 1)
            color_map = {
                RiskLevel.LOW: "#1a7f37",
                RiskLevel.MEDIUM: "#b08800",
                RiskLevel.HIGH: "#e36209",
                RiskLevel.CRITICAL: "#cf222e",
            }
            bar_color = color_map.get(r.risk_level, "#888")
            rows += (
                f'<tr><td style="{css}">{r.risk_level.value}</td>'
                f"<td>{r.supplier_count}</td>"
                f"<td>{r.percentage:.1f}%</td>"
                f'<td><div class="bar" style="width:{bar_width}%;'
                f'background:{bar_color};">&nbsp;</div></td></tr>'
            )
        if not rows:
            rows = '<tr><td colspan="4">No risk data</td></tr>'
        return (
            '<div class="section">\n<h2>3. Risk Distribution</h2>\n'
            "<table><thead><tr><th>Risk Level</th><th>Suppliers</th>"
            "<th>Percentage</th><th>Distribution</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supplier_status(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 4: Supplier Status."""
        ss = data.supplier_status
        return (
            '<div class="section">\n<h2>4. Supplier Status</h2>\n<div>\n'
            f'<div class="kpi-card"><div class="kpi-value">'
            f'{ss.total_suppliers}</div>'
            f'<div class="kpi-label">Total</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">{ss.onboarded}</div>'
            f'<div class="kpi-label">Onboarded</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">{ss.dd_complete}</div>'
            f'<div class="kpi-label">DD Complete</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">{ss.dd_pending}</div>'
            f'<div class="kpi-label">DD Pending</div></div>\n'
            f'<div class="kpi-card"><div class="kpi-value">{ss.dd_overdue}</div>'
            f'<div class="kpi-label">DD Overdue</div></div>\n'
            "</div>\n</div>"
        )

    def _html_geolocation_coverage(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 5: Geolocation Coverage."""
        gc = data.geolocation_coverage
        countries = ", ".join(gc.country_list) if gc.country_list else "N/A"
        total_plots = gc.plots_validated + gc.plots_pending + gc.plots_failed
        return (
            '<div class="section">\n<h2>5. Geolocation Coverage</h2>\n'
            "<table><tbody>"
            f"<tr><th>Total Plots</th><td>{total_plots}</td></tr>"
            f"<tr><th>Validated</th><td>{gc.plots_validated}</td></tr>"
            f"<tr><th>Pending</th><td>{gc.plots_pending}</td></tr>"
            f"<tr><th>Failed</th><td>{gc.plots_failed}</td></tr>"
            f"<tr><th>Total Area</th><td>{_fmt_area(gc.total_area_ha)}</td></tr>"
            f"<tr><th>Countries</th><td>{gc.countries_covered}</td></tr>"
            f"<tr><th>Country List</th><td>{countries}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_certification_summary(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 6: Certification Summary."""
        rows = ""
        for c in data.certifications:
            rows += (
                f"<tr><td>{c.scheme}</td><td>{c.active_count}</td>"
                f"<td>{c.expired_count}</td><td>{c.expiring_soon_count}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="4">No certifications tracked</td></tr>'
        return (
            '<div class="section">\n<h2>6. Certification Summary</h2>\n'
            "<table><thead><tr><th>Scheme</th><th>Active</th>"
            "<th>Expired</th><th>Expiring Soon</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_data_quality(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 7: Data Quality."""
        overall = data.overall_data_quality_score
        rows = ""
        for m in data.data_quality:
            color = "#1a7f37" if m.score >= 80 else (
                "#b08800" if m.score >= 60 else "#cf222e"
            )
            rows += (
                f"<tr><td>{m.category}</td>"
                f'<td style="color:{color};font-weight:bold;">{m.score:.1f}%</td>'
                f"<td>{m.records_valid}/{m.records_total}</td>"
                f"<td>{m.issues_count}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="4">No data quality metrics</td></tr>'
        return (
            '<div class="section">\n<h2>7. Data Quality Score</h2>\n'
            f"<p><strong>Overall:</strong> {overall:.1f}%</p>\n"
            "<table><thead><tr><th>Category</th><th>Score</th>"
            "<th>Records (Valid/Total)</th><th>Issues</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_upcoming_deadlines(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 8: Upcoming Deadlines."""
        rows = ""
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            owner = d.owner or "TBD"
            urgency = _deadline_urgency(d.days_remaining)
            urgency_css = (
                "color:#cf222e;font-weight:bold;" if urgency in ("OVERDUE", "URGENT")
                else ("color:#b08800;" if urgency == "SOON" else "")
            )
            rows += (
                f"<tr><td>{d.deadline_id}</td><td>{d.title}</td>"
                f"<td>{d.category}</td><td>{d.deadline_date.isoformat()}</td>"
                f'<td style="{urgency_css}">{d.days_remaining} [{urgency}]</td>'
                f"<td>{d.readiness_pct:.0f}%</td><td>{owner}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="7">No upcoming deadlines</td></tr>'
        return (
            '<div class="section">\n<h2>8. Upcoming Deadlines</h2>\n'
            "<table><thead><tr><th>ID</th><th>Title</th><th>Category</th>"
            "<th>Deadline</th><th>Days Left</th><th>Readiness</th>"
            f"<th>Owner</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recent_activity(self, data: ComplianceDashboardInput) -> str:
        """HTML Section 9: Recent Activity."""
        rows = ""
        for a in data.recent_activity:
            actor = a.actor or "System"
            ts = a.timestamp.strftime("%Y-%m-%d %H:%M")
            rows += (
                f"<tr><td>{a.event_id}</td><td>{a.event_type.value}</td>"
                f"<td>{a.description}</td><td>{ts}</td><td>{actor}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No recent activity</td></tr>'
        return (
            '<div class="section">\n<h2>9. Recent Activity</h2>\n'
            "<table><thead><tr><th>Event</th><th>Type</th>"
            "<th>Description</th><th>Time</th><th>Actor</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_provenance(self, data: ComplianceDashboardInput) -> str:
        """HTML provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            f"<p>Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} "
            f"| {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
