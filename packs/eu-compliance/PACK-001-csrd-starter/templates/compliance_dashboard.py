# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: Compliance Dashboard Template
=================================================

Generates real-time compliance dashboard data with overall score,
per-standard compliance, data completeness heatmap, outstanding
actions, historical trends, alert summary, and upcoming deadlines.

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


# =============================================================================
# ENUMS
# =============================================================================

class ComplianceLevel(str, Enum):
    """Compliance level for a standard."""
    FULL = "FULL"
    SUBSTANTIAL = "SUBSTANTIAL"
    PARTIAL = "PARTIAL"
    MINIMAL = "MINIMAL"
    NONE = "NONE"


class ActionStatus(str, Enum):
    """Status of an outstanding action."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertCategory(str, Enum):
    """Alert category classification."""
    DATA_GAP = "DATA_GAP"
    DEADLINE = "DEADLINE"
    QUALITY = "QUALITY"
    COMPLIANCE = "COMPLIANCE"
    SYSTEM = "SYSTEM"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StandardComplianceEntry(BaseModel):
    """Compliance status for a single ESRS standard on the dashboard."""
    standard_id: str = Field(..., description="ESRS standard ID (e.g. E1)")
    standard_name: str = Field(..., description="Standard name")
    compliance_pct: float = Field(0.0, ge=0.0, le=100.0, description="Compliance percentage")
    compliance_level: ComplianceLevel = Field(ComplianceLevel.NONE, description="Compliance level")
    data_points_filled: int = Field(0, ge=0, description="Data points completed")
    data_points_required: int = Field(0, ge=0, description="Data points required")
    is_material: bool = Field(True, description="Whether standard is material")
    trend: Optional[str] = Field(None, description="Trend indicator (up, down, stable)")

    def model_post_init(self, __context: Any) -> None:
        """Auto-derive compliance level from percentage if default."""
        if self.compliance_level == ComplianceLevel.NONE and self.compliance_pct > 0:
            if self.compliance_pct >= 95:
                self.compliance_level = ComplianceLevel.FULL
            elif self.compliance_pct >= 75:
                self.compliance_level = ComplianceLevel.SUBSTANTIAL
            elif self.compliance_pct >= 50:
                self.compliance_level = ComplianceLevel.PARTIAL
            elif self.compliance_pct >= 25:
                self.compliance_level = ComplianceLevel.MINIMAL
            else:
                self.compliance_level = ComplianceLevel.NONE


class DataCompletenessCell(BaseModel):
    """Single cell in the data completeness heatmap."""
    standard_id: str = Field(..., description="ESRS standard")
    data_category: str = Field(..., description="Data category (e.g. 'Metrics', 'Policies')")
    filled: int = Field(0, ge=0, description="Data points filled")
    required: int = Field(0, ge=0, description="Data points required")

    @property
    def completeness_pct(self) -> float:
        """Completeness percentage for this cell."""
        if self.required == 0:
            return 100.0
        return (self.filled / self.required) * 100.0

    @property
    def heat_level(self) -> str:
        """Heat level label for visualization."""
        pct = self.completeness_pct
        if pct >= 90:
            return "COMPLETE"
        if pct >= 70:
            return "HIGH"
        if pct >= 40:
            return "MEDIUM"
        if pct > 0:
            return "LOW"
        return "EMPTY"


class OutstandingAction(BaseModel):
    """Outstanding action item for the dashboard."""
    action_id: str = Field(..., description="Action identifier")
    title: str = Field(..., description="Action title")
    description: str = Field("", description="Action description")
    priority: str = Field("MEDIUM", description="Priority: CRITICAL, HIGH, MEDIUM, LOW")
    status: ActionStatus = Field(ActionStatus.OPEN, description="Current status")
    owner: Optional[str] = Field(None, description="Assigned owner")
    deadline: Optional[date] = Field(None, description="Due date")
    esrs_reference: Optional[str] = Field(None, description="Related ESRS standard")
    days_remaining: Optional[int] = Field(None, description="Days until deadline")
    completion_pct: float = Field(0.0, ge=0.0, le=100.0, description="Completion %")


class ComplianceTrendPoint(BaseModel):
    """Quarterly compliance trend data point."""
    period_label: str = Field(..., description="Period label (e.g. 'Q1 2025')")
    period_date: date = Field(..., description="Period end date")
    overall_score_pct: float = Field(0.0, ge=0.0, le=100.0, description="Overall compliance %")
    data_completeness_pct: float = Field(0.0, ge=0.0, le=100.0, description="Data completeness %")
    actions_closed: int = Field(0, ge=0, description="Actions closed in period")
    actions_opened: int = Field(0, ge=0, description="Actions opened in period")


class AlertEntry(BaseModel):
    """Dashboard alert entry."""
    alert_id: str = Field(..., description="Alert identifier")
    severity: AlertSeverity = Field(..., description="Severity level")
    category: AlertCategory = Field(..., description="Alert category")
    title: str = Field(..., description="Alert title")
    message: str = Field("", description="Alert message body")
    esrs_reference: Optional[str] = Field(None, description="Related ESRS standard")
    created_at: Optional[datetime] = Field(None, description="Alert creation time")
    is_acknowledged: bool = Field(False, description="Whether alert has been acknowledged")


class UpcomingDeadline(BaseModel):
    """Upcoming regulatory or internal deadline."""
    deadline_id: str = Field(..., description="Deadline identifier")
    title: str = Field(..., description="Deadline title")
    deadline_date: date = Field(..., description="Due date")
    regulation: str = Field("", description="Regulation name")
    description: str = Field("", description="What is due")
    days_remaining: int = Field(0, description="Days until deadline")
    readiness_pct: float = Field(0.0, ge=0.0, le=100.0, description="Readiness %")
    owner: Optional[str] = Field(None, description="Responsible party")


class ComplianceDashboardInput(BaseModel):
    """Full input for the compliance dashboard."""
    company_name: str = Field(..., description="Reporting entity")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Fiscal year")
    dashboard_date: date = Field(default_factory=date.today, description="Dashboard snapshot date")
    overall_compliance_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall compliance score"
    )
    standard_compliance: List[StandardComplianceEntry] = Field(
        default_factory=list, description="Per-standard compliance"
    )
    data_completeness: List[DataCompletenessCell] = Field(
        default_factory=list, description="Data completeness heatmap cells"
    )
    outstanding_actions: List[OutstandingAction] = Field(
        default_factory=list, description="Outstanding actions"
    )
    compliance_trends: List[ComplianceTrendPoint] = Field(
        default_factory=list, description="Historical compliance trends"
    )
    alerts: List[AlertEntry] = Field(default_factory=list, description="Active alerts")
    upcoming_deadlines: List[UpcomingDeadline] = Field(
        default_factory=list, description="Upcoming deadlines"
    )

    @property
    def critical_alerts(self) -> int:
        """Count of unacknowledged critical alerts."""
        return sum(
            1 for a in self.alerts
            if a.severity == AlertSeverity.CRITICAL and not a.is_acknowledged
        )

    @property
    def overdue_actions(self) -> int:
        """Count of overdue actions."""
        return sum(1 for a in self.outstanding_actions if a.status == ActionStatus.OVERDUE)

    @property
    def data_completeness_overall(self) -> float:
        """Overall data completeness percentage."""
        total_filled = sum(c.filled for c in self.data_completeness)
        total_required = sum(c.required for c in self.data_completeness)
        if total_required == 0:
            return 0.0
        return (total_filled / total_required) * 100.0


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _level_badge(level: ComplianceLevel) -> str:
    """Text badge for compliance level."""
    return f"[{level.value}]"


def _level_css(level: ComplianceLevel) -> str:
    """CSS class for compliance level."""
    return f"level-{level.value.lower()}"


def _severity_badge(severity: AlertSeverity) -> str:
    """Text badge for alert severity."""
    return f"[{severity.value}]"


def _severity_css(severity: AlertSeverity) -> str:
    """CSS class for severity."""
    return f"severity-{severity.value.lower()}"


def _status_badge(status: ActionStatus) -> str:
    """Text badge for action status."""
    return f"[{status.value}]"


def _heat_css(heat_level: str) -> str:
    """CSS class for heatmap cell."""
    return f"heat-{heat_level.lower()}"


def _priority_sort(priority: str) -> int:
    """Numeric sort key for priority strings."""
    return {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(priority.upper(), 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ComplianceDashboardTemplate:
    """Generate compliance status dashboard data.

    Widgets:
        1. Overall Compliance Score (0-100%)
        2. Per-Standard Compliance (E1-E5, S1-S4, G1)
        3. Data Completeness Heatmap (data points filled vs required)
        4. Outstanding Actions (priority, deadline, owner)
        5. Historical Compliance Trends (quarterly over 2 years)
        6. Alert Summary (critical, warning, info)
        7. Upcoming Deadlines

    Example:
        >>> template = ComplianceDashboardTemplate()
        >>> data = ComplianceDashboardInput(company_name="Acme", reporting_year=2025, ...)
        >>> md = template.render_markdown(data)
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
            data: Validated dashboard input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_standard_compliance(data),
            self._md_data_completeness(data),
            self._md_outstanding_actions(data),
            self._md_compliance_trends(data),
            self._md_alerts(data),
            self._md_deadlines(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ComplianceDashboardInput) -> str:
        """Render the dashboard as HTML.

        Args:
            data: Validated dashboard input.

        Returns:
            HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_overall_score(data),
            self._html_standard_compliance(data),
            self._html_data_completeness(data),
            self._html_outstanding_actions(data),
            self._html_compliance_trends(data),
            self._html_alerts(data),
            self._html_deadlines(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: ComplianceDashboardInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated dashboard input.

        Returns:
            Dictionary for serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)

        return {
            "template": "compliance_dashboard",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "dashboard_date": data.dashboard_date.isoformat(),
            "widgets": {
                "overall_score": {
                    "score_pct": data.overall_compliance_score,
                    "critical_alerts": data.critical_alerts,
                    "overdue_actions": data.overdue_actions,
                    "data_completeness_pct": data.data_completeness_overall,
                },
                "standard_compliance": [
                    s.model_dump(mode="json") for s in data.standard_compliance
                ],
                "data_completeness": [
                    {**c.model_dump(mode="json"), "completeness_pct": c.completeness_pct,
                     "heat_level": c.heat_level}
                    for c in data.data_completeness
                ],
                "outstanding_actions": [
                    a.model_dump(mode="json") for a in data.outstanding_actions
                ],
                "compliance_trends": [
                    t.model_dump(mode="json") for t in data.compliance_trends
                ],
                "alerts": [a.model_dump(mode="json") for a in data.alerts],
                "upcoming_deadlines": [
                    d.model_dump(mode="json") for d in data.upcoming_deadlines
                ],
            },
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: ComplianceDashboardInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ComplianceDashboardInput) -> str:
        """Markdown header."""
        return (
            f"# CSRD Compliance Dashboard - {data.company_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Snapshot Date:** {data.dashboard_date.isoformat()}\n\n---"
        )

    def _md_overall_score(self, data: ComplianceDashboardInput) -> str:
        """Overall compliance score widget."""
        return (
            "## 1. Overall Compliance Score\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Overall Compliance | **{data.overall_compliance_score:.1f}%** |\n"
            f"| Data Completeness | {data.data_completeness_overall:.1f}% |\n"
            f"| Critical Alerts | {data.critical_alerts} |\n"
            f"| Overdue Actions | {data.overdue_actions} |"
        )

    def _md_standard_compliance(self, data: ComplianceDashboardInput) -> str:
        """Per-standard compliance table."""
        lines = [
            "## 2. Per-Standard Compliance",
            "",
            "| Standard | Name | Score | Level | Data Points | Material | Trend |",
            "|----------|------|-------|-------|-------------|----------|-------|",
        ]
        for s in data.standard_compliance:
            dp = f"{s.data_points_filled}/{s.data_points_required}"
            mat = "Yes" if s.is_material else "No"
            trend = s.trend or "-"
            lines.append(
                f"| {s.standard_id} | {s.standard_name} | {s.compliance_pct:.0f}% "
                f"| {_level_badge(s.compliance_level)} | {dp} | {mat} | {trend} |"
            )
        if not data.standard_compliance:
            lines.append("| - | No standards tracked | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_data_completeness(self, data: ComplianceDashboardInput) -> str:
        """Data completeness heatmap as a table."""
        if not data.data_completeness:
            return "## 3. Data Completeness Heatmap\n\nNo data completeness information."
        standards = sorted(set(c.standard_id for c in data.data_completeness))
        categories = sorted(set(c.data_category for c in data.data_completeness))
        lookup: Dict[tuple, DataCompletenessCell] = {}
        for c in data.data_completeness:
            lookup[(c.standard_id, c.data_category)] = c

        header = "| Standard | " + " | ".join(categories) + " |"
        separator = "|----------|" + "|".join("-" * (len(cat) + 2) for cat in categories) + "|"
        lines = [
            "## 3. Data Completeness Heatmap",
            "",
            header,
            separator,
        ]
        for std in standards:
            cells = []
            for cat in categories:
                cell = lookup.get((std, cat))
                if cell:
                    cells.append(f"{cell.completeness_pct:.0f}% ({cell.heat_level})")
                else:
                    cells.append("-")
            lines.append(f"| {std} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _md_outstanding_actions(self, data: ComplianceDashboardInput) -> str:
        """Outstanding actions table."""
        sorted_actions = sorted(
            data.outstanding_actions,
            key=lambda a: (_priority_sort(a.priority), a.deadline or date.max),
        )
        lines = [
            "## 4. Outstanding Actions",
            "",
            f"**Total:** {len(data.outstanding_actions)} | "
            f"**Overdue:** {data.overdue_actions}",
            "",
            "| ID | Priority | Title | Status | Owner | Deadline | Days Left | Progress |",
            "|----|----------|-------|--------|-------|----------|-----------|----------|",
        ]
        for a in sorted_actions:
            owner = a.owner or "Unassigned"
            deadline = a.deadline.isoformat() if a.deadline else "TBD"
            days = str(a.days_remaining) if a.days_remaining is not None else "N/A"
            lines.append(
                f"| {a.action_id} | {a.priority} | {a.title} "
                f"| {_status_badge(a.status)} | {owner} | {deadline} "
                f"| {days} | {a.completion_pct:.0f}% |"
            )
        if not data.outstanding_actions:
            lines.append("| - | - | No outstanding actions | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_compliance_trends(self, data: ComplianceDashboardInput) -> str:
        """Historical compliance trends."""
        if not data.compliance_trends:
            return "## 5. Historical Compliance Trends\n\nNo trend data available."
        lines = [
            "## 5. Historical Compliance Trends",
            "",
            "| Period | Compliance | Data Completeness | Actions Closed | Actions Opened |",
            "|--------|-----------|-------------------|----------------|----------------|",
        ]
        for t in sorted(data.compliance_trends, key=lambda x: x.period_date):
            lines.append(
                f"| {t.period_label} | {t.overall_score_pct:.1f}% "
                f"| {t.data_completeness_pct:.1f}% | {t.actions_closed} "
                f"| {t.actions_opened} |"
            )
        return "\n".join(lines)

    def _md_alerts(self, data: ComplianceDashboardInput) -> str:
        """Alert summary."""
        critical = sum(1 for a in data.alerts if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in data.alerts if a.severity == AlertSeverity.WARNING)
        info = sum(1 for a in data.alerts if a.severity == AlertSeverity.INFO)
        lines = [
            "## 6. Alert Summary",
            "",
            f"**Critical:** {critical} | **Warning:** {warning} | **Info:** {info}",
            "",
            "| ID | Severity | Category | Title | Message | ESRS Ref | Ack |",
            "|----|----------|----------|-------|---------|----------|-----|",
        ]
        sorted_alerts = sorted(
            data.alerts,
            key=lambda a: (
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(a.severity.value, 99),
                a.is_acknowledged,
            ),
        )
        for a in sorted_alerts:
            ref = a.esrs_reference or "-"
            ack = "Yes" if a.is_acknowledged else "No"
            lines.append(
                f"| {a.alert_id} | {_severity_badge(a.severity)} | {a.category.value} "
                f"| {a.title} | {a.message} | {ref} | {ack} |"
            )
        if not data.alerts:
            lines.append("| - | - | - | No active alerts | - | - | - |")
        return "\n".join(lines)

    def _md_deadlines(self, data: ComplianceDashboardInput) -> str:
        """Upcoming deadlines table."""
        lines = [
            "## 7. Upcoming Deadlines",
            "",
            "| ID | Title | Regulation | Deadline | Days Left | Readiness | Owner |",
            "|----|-------|-----------|----------|-----------|-----------|-------|",
        ]
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            owner = d.owner or "TBD"
            lines.append(
                f"| {d.deadline_id} | {d.title} | {d.regulation} "
                f"| {d.deadline_date.isoformat()} | {d.days_remaining} "
                f"| {d.readiness_pct:.0f}% | {owner} |"
            )
        if not data.upcoming_deadlines:
            lines.append("| - | - | No upcoming deadlines | - | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: ComplianceDashboardInput) -> str:
        """Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """HTML wrapper."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Compliance Dashboard - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;max-width:1200px;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".score-card{display:inline-block;text-align:center;padding:1rem 2rem;"
            "border:2px solid #ccc;border-radius:8px;margin:0.5rem;}\n"
            ".score-value{font-size:2rem;font-weight:bold;}\n"
            ".score-label{font-size:0.85rem;color:#666;}\n"
            ".level-full{color:#1a7f37;font-weight:bold;}\n"
            ".level-substantial{color:#2da44e;}\n"
            ".level-partial{color:#b08800;}\n"
            ".level-minimal{color:#cf222e;}\n"
            ".level-none{color:#888;}\n"
            ".severity-critical{color:#cf222e;font-weight:bold;}\n"
            ".severity-warning{color:#b08800;font-weight:bold;}\n"
            ".severity-info{color:#0969da;}\n"
            ".heat-complete{background:#d1fae5;}\n"
            ".heat-high{background:#fef3c7;}\n"
            ".heat-medium{background:#fed7aa;}\n"
            ".heat-low{background:#fecaca;}\n"
            ".heat-empty{background:#f3f4f6;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: ComplianceDashboardInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>CSRD Compliance Dashboard &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Snapshot:</strong> {data.dashboard_date.isoformat()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_score(self, data: ComplianceDashboardInput) -> str:
        """HTML overall score cards."""
        return (
            '<div class="section">\n<h2>1. Overall Compliance Score</h2>\n<div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{data.overall_compliance_score:.0f}%</div>'
            f'<div class="score-label">Compliance</div></div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{data.data_completeness_overall:.0f}%</div>'
            f'<div class="score-label">Data Completeness</div></div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{data.critical_alerts}</div>'
            f'<div class="score-label">Critical Alerts</div></div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{data.overdue_actions}</div>'
            f'<div class="score-label">Overdue Actions</div></div>\n'
            "</div>\n</div>"
        )

    def _html_standard_compliance(self, data: ComplianceDashboardInput) -> str:
        """HTML per-standard compliance table."""
        rows = []
        for s in data.standard_compliance:
            css = _level_css(s.compliance_level)
            dp = f"{s.data_points_filled}/{s.data_points_required}"
            mat = "Yes" if s.is_material else "No"
            trend = s.trend or "-"
            rows.append(
                f"<tr><td>{s.standard_id}</td><td>{s.standard_name}</td>"
                f"<td>{s.compliance_pct:.0f}%</td>"
                f'<td class="{css}">{s.compliance_level.value}</td>'
                f"<td>{dp}</td><td>{mat}</td><td>{trend}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="7">No standards tracked</td></tr>')
        return (
            '<div class="section">\n<h2>2. Per-Standard Compliance</h2>\n'
            "<table><thead><tr><th>Standard</th><th>Name</th><th>Score</th>"
            "<th>Level</th><th>Data Points</th><th>Material</th>"
            f"<th>Trend</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_data_completeness(self, data: ComplianceDashboardInput) -> str:
        """HTML data completeness heatmap."""
        if not data.data_completeness:
            return (
                '<div class="section"><h2>3. Data Completeness</h2>'
                "<p>No data.</p></div>"
            )
        standards = sorted(set(c.standard_id for c in data.data_completeness))
        categories = sorted(set(c.data_category for c in data.data_completeness))
        lookup: Dict[tuple, DataCompletenessCell] = {}
        for c in data.data_completeness:
            lookup[(c.standard_id, c.data_category)] = c

        header_cells = "".join(f"<th>{cat}</th>" for cat in categories)
        rows = []
        for std in standards:
            cells = []
            for cat in categories:
                cell = lookup.get((std, cat))
                if cell:
                    css = _heat_css(cell.heat_level)
                    cells.append(f'<td class="{css}">{cell.completeness_pct:.0f}%</td>')
                else:
                    cells.append("<td>-</td>")
            rows.append(f"<tr><td>{std}</td>{''.join(cells)}</tr>")

        return (
            '<div class="section">\n<h2>3. Data Completeness Heatmap</h2>\n'
            f"<table><thead><tr><th>Standard</th>{header_cells}</tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_outstanding_actions(self, data: ComplianceDashboardInput) -> str:
        """HTML outstanding actions table."""
        sorted_actions = sorted(
            data.outstanding_actions,
            key=lambda a: (_priority_sort(a.priority), a.deadline or date.max),
        )
        rows = []
        for a in sorted_actions:
            owner = a.owner or "Unassigned"
            deadline = a.deadline.isoformat() if a.deadline else "TBD"
            days = str(a.days_remaining) if a.days_remaining is not None else "N/A"
            rows.append(
                f"<tr><td>{a.action_id}</td><td>{a.priority}</td>"
                f"<td>{a.title}</td><td>{a.status.value}</td>"
                f"<td>{owner}</td><td>{deadline}</td><td>{days}</td>"
                f"<td>{a.completion_pct:.0f}%</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="8">No outstanding actions</td></tr>')
        return (
            '<div class="section">\n<h2>4. Outstanding Actions</h2>\n'
            f"<p><strong>Total:</strong> {len(data.outstanding_actions)} | "
            f"<strong>Overdue:</strong> {data.overdue_actions}</p>\n"
            "<table><thead><tr><th>ID</th><th>Priority</th><th>Title</th>"
            "<th>Status</th><th>Owner</th><th>Deadline</th><th>Days</th>"
            f"<th>Progress</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_compliance_trends(self, data: ComplianceDashboardInput) -> str:
        """HTML compliance trends."""
        if not data.compliance_trends:
            return ""
        rows = []
        for t in sorted(data.compliance_trends, key=lambda x: x.period_date):
            rows.append(
                f"<tr><td>{t.period_label}</td><td>{t.overall_score_pct:.1f}%</td>"
                f"<td>{t.data_completeness_pct:.1f}%</td>"
                f"<td>{t.actions_closed}</td><td>{t.actions_opened}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>5. Historical Trends</h2>\n'
            "<table><thead><tr><th>Period</th><th>Compliance</th>"
            "<th>Data Completeness</th><th>Closed</th><th>Opened</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_alerts(self, data: ComplianceDashboardInput) -> str:
        """HTML alert summary."""
        sorted_alerts = sorted(
            data.alerts,
            key=lambda a: (
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(a.severity.value, 99),
                a.is_acknowledged,
            ),
        )
        rows = []
        for a in sorted_alerts:
            css = _severity_css(a.severity)
            ref = a.esrs_reference or "-"
            ack = "Yes" if a.is_acknowledged else "No"
            rows.append(
                f"<tr><td>{a.alert_id}</td>"
                f'<td class="{css}">{a.severity.value}</td>'
                f"<td>{a.category.value}</td><td>{a.title}</td>"
                f"<td>{a.message}</td><td>{ref}</td><td>{ack}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="7">No active alerts</td></tr>')
        critical = sum(1 for a in data.alerts if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in data.alerts if a.severity == AlertSeverity.WARNING)
        info = sum(1 for a in data.alerts if a.severity == AlertSeverity.INFO)
        return (
            '<div class="section">\n<h2>6. Alert Summary</h2>\n'
            f"<p><strong>Critical:</strong> {critical} | "
            f"<strong>Warning:</strong> {warning} | "
            f"<strong>Info:</strong> {info}</p>\n"
            "<table><thead><tr><th>ID</th><th>Severity</th><th>Category</th>"
            "<th>Title</th><th>Message</th><th>ESRS</th><th>Ack</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_deadlines(self, data: ComplianceDashboardInput) -> str:
        """HTML upcoming deadlines."""
        rows = []
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            owner = d.owner or "TBD"
            rows.append(
                f"<tr><td>{d.deadline_id}</td><td>{d.title}</td>"
                f"<td>{d.regulation}</td><td>{d.deadline_date.isoformat()}</td>"
                f"<td>{d.days_remaining}</td><td>{d.readiness_pct:.0f}%</td>"
                f"<td>{owner}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="7">No upcoming deadlines</td></tr>')
        return (
            '<div class="section">\n<h2>7. Upcoming Deadlines</h2>\n'
            "<table><thead><tr><th>ID</th><th>Title</th><th>Regulation</th>"
            "<th>Deadline</th><th>Days</th><th>Readiness</th>"
            f"<th>Owner</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: ComplianceDashboardInput) -> str:
        """HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
