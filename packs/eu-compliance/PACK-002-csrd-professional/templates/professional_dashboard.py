# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Professional Dashboard Template
===================================================

Enhanced real-time compliance dashboard template with ESRS standard
compliance grid, quality gate status, approval pipeline, regulatory
alerts, benchmark position, entity status, SLO adherence, and
deadline tracking.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
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

class GateOutcome(str, Enum):
    """Quality gate outcome."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class ApprovalLevel(str, Enum):
    """Approval pipeline level."""
    DATA_ENTRY = "DATA_ENTRY"
    REVIEWER = "REVIEWER"
    MANAGER = "MANAGER"
    DIRECTOR = "DIRECTOR"
    EXECUTIVE = "EXECUTIVE"


class AlertSeverity(str, Enum):
    """Regulatory alert severity."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class TrendDirection(str, Enum):
    """Trend direction."""
    UP = "UP"
    DOWN = "DOWN"
    STABLE = "STABLE"


class SLOStatusLevel(str, Enum):
    """SLO adherence status."""
    MEETING = "MEETING"
    AT_RISK = "AT_RISK"
    BREACHED = "BREACHED"


class DeadlineStatus(str, Enum):
    """Deadline tracking status."""
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    OVERDUE = "OVERDUE"
    COMPLETED = "COMPLETED"


class CalculationStatus(str, Enum):
    """Calculation pipeline status."""
    COMPLETE = "COMPLETE"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING = "PENDING"
    ERROR = "ERROR"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StandardCompliance(BaseModel):
    """Compliance data for a single ESRS standard."""
    standard_id: str = Field(..., description="ESRS standard ID")
    standard_name: str = Field("", description="Standard name")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage %")
    rule_pass_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Validation rule pass rate %"
    )
    data_quality: float = Field(
        0.0, ge=0.0, le=100.0, description="Data quality score %"
    )


class QualityGateStatus(BaseModel):
    """Quality gate evaluation status."""
    gate_name: str = Field(..., description="Gate name")
    score: float = Field(..., ge=0.0, le=100.0, description="Gate score")
    threshold: float = Field(..., ge=0.0, le=100.0, description="Pass threshold")
    passed: bool = Field(False, description="Whether gate passed")
    last_evaluated: Optional[datetime] = Field(
        None, description="Last evaluation time"
    )


class ApprovalPipelineStatus(BaseModel):
    """Approval pipeline status."""
    current_level: ApprovalLevel = Field(
        ApprovalLevel.DATA_ENTRY, description="Current approval level"
    )
    pending_count: int = Field(0, ge=0, description="Pending approvals")
    avg_approval_hours: float = Field(
        0.0, ge=0.0, description="Average approval time in hours"
    )
    total_approved: int = Field(0, ge=0, description="Total approved items")
    total_rejected: int = Field(0, ge=0, description="Total rejected items")


class RegulatoryAlert(BaseModel):
    """Regulatory change alert."""
    title: str = Field(..., description="Alert title")
    severity: AlertSeverity = Field(..., description="Alert severity")
    affected_standards: List[str] = Field(
        default_factory=list, description="Affected ESRS standards"
    )
    alert_date: date = Field(..., description="Alert date")
    description: Optional[str] = Field(None, description="Alert description")
    action_required: Optional[str] = Field(None, description="Required action")


class BenchmarkPosition(BaseModel):
    """Benchmark position indicator."""
    overall_percentile: float = Field(
        ..., ge=0.0, le=100.0, description="Overall percentile rank"
    )
    trend_direction: TrendDirection = Field(
        TrendDirection.STABLE, description="Trend direction"
    )
    peer_count: int = Field(0, ge=0, description="Number of peers in comparison")
    sector: Optional[str] = Field(None, description="Sector name")


class EntityComplianceStatus(BaseModel):
    """Compliance status for a single entity."""
    entity_name: str = Field(..., description="Entity name")
    data_completeness: float = Field(
        0.0, ge=0.0, le=100.0, description="Data completeness %"
    )
    calculation_status: CalculationStatus = Field(
        CalculationStatus.PENDING, description="Calculation pipeline status"
    )
    approval_status: str = Field("PENDING", description="Approval status")


class SLOStatus(BaseModel):
    """SLO adherence status."""
    slo_name: str = Field(..., description="SLO name")
    target: float = Field(..., ge=0.0, le=100.0, description="SLO target %")
    current: float = Field(0.0, ge=0.0, le=100.0, description="Current value %")
    status: SLOStatusLevel = Field(SLOStatusLevel.MEETING, description="SLO status")


class DeadlineEntry(BaseModel):
    """Upcoming deadline entry."""
    deadline_name: str = Field(..., description="Deadline name")
    deadline_date: date = Field(..., description="Deadline date")
    days_remaining: int = Field(0, description="Days remaining")
    status: DeadlineStatus = Field(
        DeadlineStatus.ON_TRACK, description="Deadline status"
    )
    owner: Optional[str] = Field(None, description="Responsible party")


class ProfessionalDashboardInput(BaseModel):
    """Complete input for the professional dashboard."""
    organization_name: str = Field(..., description="Organization name")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Dashboard timestamp"
    )
    reporting_year: int = Field(
        default_factory=lambda: date.today().year,
        ge=2020, le=2100,
        description="Reporting year",
    )
    compliance_by_standard: Dict[str, StandardCompliance] = Field(
        default_factory=dict, description="Compliance by ESRS standard"
    )
    quality_gate_status: List[QualityGateStatus] = Field(
        default_factory=list, description="Quality gate statuses"
    )
    approval_status: ApprovalPipelineStatus = Field(
        default_factory=ApprovalPipelineStatus,
        description="Approval pipeline status",
    )
    regulatory_alerts: List[RegulatoryAlert] = Field(
        default_factory=list, description="Regulatory change alerts"
    )
    benchmark_position: Optional[BenchmarkPosition] = Field(
        None, description="Benchmark position"
    )
    entity_status: List[EntityComplianceStatus] = Field(
        default_factory=list, description="Per-entity compliance status"
    )
    slo_adherence: List[SLOStatus] = Field(
        default_factory=list, description="SLO adherence status"
    )
    upcoming_deadlines: List[DeadlineEntry] = Field(
        default_factory=list, description="Upcoming deadlines"
    )

    @property
    def overall_compliance_pct(self) -> float:
        """Calculate overall compliance percentage."""
        if not self.compliance_by_standard:
            return 0.0
        values = [s.coverage_pct for s in self.compliance_by_standard.values()]
        return sum(values) / len(values)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _gate_badge(passed: bool) -> str:
    """Badge for gate outcome."""
    return "[PASS]" if passed else "[FAIL]"


def _severity_sort(severity: AlertSeverity) -> int:
    """Sort key for severity."""
    return {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.HIGH: 1,
        AlertSeverity.MEDIUM: 2,
        AlertSeverity.LOW: 3,
        AlertSeverity.INFO: 4,
    }.get(severity, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ProfessionalDashboardTemplate:
    """Generate enhanced real-time compliance dashboard.

    Sections:
        1. Overall Compliance Score
        2. ESRS Standard Compliance Grid
        3. Quality Gate Status Panel
        4. Approval Pipeline
        5. Regulatory Change Alerts
        6. Benchmark Position
        7. Entity Status Matrix
        8. SLO Adherence
        9. Deadline Tracker
        10. Last Updated Timestamp

    Example:
        >>> template = ProfessionalDashboardTemplate()
        >>> data = ProfessionalDashboardInput(organization_name="Acme")
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "professional_dashboard"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the professional dashboard template."""
        self._render_timestamp: Optional[datetime] = None

    def render_markdown(self, data: ProfessionalDashboardInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_standard_grid(data),
            self._md_quality_gates(data),
            self._md_approval_pipeline(data),
            self._md_regulatory_alerts(data),
            self._md_benchmark_position(data),
            self._md_entity_status(data),
            self._md_slo_adherence(data),
            self._md_deadlines(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: ProfessionalDashboardInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_overall_score(data),
            self._html_standard_grid(data),
            self._html_quality_gates(data),
            self._html_approval_pipeline(data),
            self._html_regulatory_alerts(data),
            self._html_benchmark_position(data),
            self._html_entity_status(data),
            self._html_slo_adherence(data),
            self._html_deadlines(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: ProfessionalDashboardInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "overall_compliance_pct": data.overall_compliance_pct,
            "compliance_by_standard": {
                k: v.model_dump(mode="json")
                for k, v in data.compliance_by_standard.items()
            },
            "quality_gate_status": [
                g.model_dump(mode="json") for g in data.quality_gate_status
            ],
            "approval_status": data.approval_status.model_dump(mode="json"),
            "regulatory_alerts": [
                a.model_dump(mode="json") for a in data.regulatory_alerts
            ],
            "benchmark_position": (
                data.benchmark_position.model_dump(mode="json")
                if data.benchmark_position else None
            ),
            "entity_status": [
                e.model_dump(mode="json") for e in data.entity_status
            ],
            "slo_adherence": [
                s.model_dump(mode="json") for s in data.slo_adherence
            ],
            "upcoming_deadlines": [
                d.model_dump(mode="json") for d in data.upcoming_deadlines
            ],
        }

    def _compute_provenance(self, data: ProfessionalDashboardInput) -> str:
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: ProfessionalDashboardInput) -> str:
        return (
            f"# CSRD Professional Dashboard - {data.organization_name}\n"
            f"**Year:** {data.reporting_year} | "
            f"**Last Updated:** {data.timestamp.isoformat()}\n\n---"
        )

    def _md_overall_score(self, data: ProfessionalDashboardInput) -> str:
        score = data.overall_compliance_pct
        gates_passed = sum(1 for g in data.quality_gate_status if g.passed)
        gates_total = len(data.quality_gate_status)
        alerts_critical = sum(
            1 for a in data.regulatory_alerts if a.severity == AlertSeverity.CRITICAL
        )
        return (
            "## 1. Overall Compliance Score\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| **Overall Compliance** | **{score:.0f}%** |\n"
            f"| Quality Gates Passed | {gates_passed}/{gates_total} |\n"
            f"| Pending Approvals | {data.approval_status.pending_count} |\n"
            f"| Critical Alerts | {alerts_critical} |"
        )

    def _md_standard_grid(self, data: ProfessionalDashboardInput) -> str:
        if not data.compliance_by_standard:
            return "## 2. ESRS Standard Compliance\n\nNo standard data available."
        lines = [
            "## 2. ESRS Standard Compliance Grid",
            "",
            "| Standard | Name | Coverage | Rule Pass | Data Quality |",
            "|----------|------|----------|-----------|-------------|",
        ]
        for std_id in sorted(data.compliance_by_standard.keys()):
            sc = data.compliance_by_standard[std_id]
            name = sc.standard_name or std_id
            lines.append(
                f"| {sc.standard_id} | {name} "
                f"| {_fmt_pct(sc.coverage_pct)} | {_fmt_pct(sc.rule_pass_pct)} "
                f"| {_fmt_pct(sc.data_quality)} |"
            )
        return "\n".join(lines)

    def _md_quality_gates(self, data: ProfessionalDashboardInput) -> str:
        if not data.quality_gate_status:
            return "## 3. Quality Gate Status\n\nNo quality gates configured."
        lines = [
            "## 3. Quality Gate Status",
            "",
            "| Gate | Score | Threshold | Result | Last Evaluated |",
            "|------|-------|-----------|--------|----------------|",
        ]
        for g in data.quality_gate_status:
            last = g.last_evaluated.isoformat() if g.last_evaluated else "N/A"
            lines.append(
                f"| {g.gate_name} | {g.score:.1f}% | {g.threshold:.1f}% "
                f"| {_gate_badge(g.passed)} | {last} |"
            )
        return "\n".join(lines)

    def _md_approval_pipeline(self, data: ProfessionalDashboardInput) -> str:
        ap = data.approval_status
        return (
            "## 4. Approval Pipeline\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Current Level | {ap.current_level.value} |\n"
            f"| Pending Approvals | {ap.pending_count} |\n"
            f"| Avg Approval Time | {ap.avg_approval_hours:.1f} hours |\n"
            f"| Total Approved | {ap.total_approved} |\n"
            f"| Total Rejected | {ap.total_rejected} |"
        )

    def _md_regulatory_alerts(self, data: ProfessionalDashboardInput) -> str:
        if not data.regulatory_alerts:
            return "## 5. Regulatory Alerts\n\nNo active alerts."
        sorted_alerts = sorted(
            data.regulatory_alerts, key=lambda a: _severity_sort(a.severity)
        )
        lines = [
            "## 5. Regulatory Change Alerts",
            "",
            "| Severity | Title | Standards | Date | Action |",
            "|----------|-------|-----------|------|--------|",
        ]
        for a in sorted_alerts:
            stds = ", ".join(a.affected_standards) if a.affected_standards else "-"
            action = a.action_required or "-"
            lines.append(
                f"| [{a.severity.value}] | {a.title} "
                f"| {stds} | {a.alert_date.isoformat()} | {action} |"
            )
        return "\n".join(lines)

    def _md_benchmark_position(self, data: ProfessionalDashboardInput) -> str:
        if not data.benchmark_position:
            return "## 6. Benchmark Position\n\nNo benchmark data available."
        bp = data.benchmark_position
        sector = bp.sector or "N/A"
        return (
            "## 6. Benchmark Position\n\n"
            f"- **Overall Percentile:** {bp.overall_percentile:.0f}th\n"
            f"- **Trend:** {bp.trend_direction.value}\n"
            f"- **Peer Count:** {bp.peer_count}\n"
            f"- **Sector:** {sector}"
        )

    def _md_entity_status(self, data: ProfessionalDashboardInput) -> str:
        if not data.entity_status:
            return "## 7. Entity Status\n\nNo entity data."
        lines = [
            "## 7. Entity Status Matrix",
            "",
            "| Entity | Data Completeness | Calculation | Approval |",
            "|--------|------------------|-------------|----------|",
        ]
        for e in data.entity_status:
            lines.append(
                f"| {e.entity_name} | {_fmt_pct(e.data_completeness)} "
                f"| {e.calculation_status.value} | {e.approval_status} |"
            )
        return "\n".join(lines)

    def _md_slo_adherence(self, data: ProfessionalDashboardInput) -> str:
        if not data.slo_adherence:
            return "## 8. SLO Adherence\n\nNo SLOs configured."
        lines = [
            "## 8. SLO Adherence",
            "",
            "| SLO | Target | Current | Status |",
            "|-----|--------|---------|--------|",
        ]
        for s in data.slo_adherence:
            lines.append(
                f"| {s.slo_name} | {_fmt_pct(s.target)} "
                f"| {_fmt_pct(s.current)} | [{s.status.value}] |"
            )
        return "\n".join(lines)

    def _md_deadlines(self, data: ProfessionalDashboardInput) -> str:
        if not data.upcoming_deadlines:
            return "## 9. Deadline Tracker\n\nNo upcoming deadlines."
        lines = [
            "## 9. Deadline Tracker",
            "",
            "| Deadline | Date | Days Left | Status | Owner |",
            "|----------|------|-----------|--------|-------|",
        ]
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            owner = d.owner or "TBD"
            lines.append(
                f"| {d.deadline_name} | {d.deadline_date.isoformat()} "
                f"| {d.days_remaining} | [{d.status.value}] | {owner} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: ProfessionalDashboardInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Professional Dashboard - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1400px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".big-score{font-size:3rem;font-weight:bold;color:#0f3460;text-align:center;"
            "padding:1rem;}\n"
            ".score-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".score-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".score-label{font-size:0.85rem;color:#666;}\n"
            ".gate-pass{color:#1a7f37;font-weight:bold;}\n"
            ".gate-fail{color:#cf222e;font-weight:bold;}\n"
            ".severity-critical{color:#cf222e;font-weight:bold;}\n"
            ".severity-high{color:#e36209;font-weight:bold;}\n"
            ".severity-medium{color:#b08800;}\n"
            ".severity-low{color:#1a7f37;}\n"
            ".slo-meeting{color:#1a7f37;font-weight:bold;}\n"
            ".slo-at-risk{color:#b08800;font-weight:bold;}\n"
            ".slo-breached{color:#cf222e;font-weight:bold;}\n"
            ".deadline-on-track{color:#1a7f37;}\n"
            ".deadline-at-risk{color:#b08800;}\n"
            ".deadline-overdue{color:#cf222e;font-weight:bold;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: ProfessionalDashboardInput) -> str:
        return (
            '<div class="section">\n'
            f"<h1>CSRD Professional Dashboard &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Year:</strong> {data.reporting_year} | "
            f"<strong>Last Updated:</strong> {data.timestamp.isoformat()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_score(self, data: ProfessionalDashboardInput) -> str:
        score = data.overall_compliance_pct
        gates_passed = sum(1 for g in data.quality_gate_status if g.passed)
        gates_total = len(data.quality_gate_status)
        alerts = sum(
            1 for a in data.regulatory_alerts if a.severity == AlertSeverity.CRITICAL
        )
        cards = [
            (f"{gates_passed}/{gates_total}", "Gates Passed"),
            (str(data.approval_status.pending_count), "Pending Approvals"),
            (str(alerts), "Critical Alerts"),
        ]
        card_html = "\n".join(
            f'<div class="score-card"><div class="score-value">{v}</div>'
            f'<div class="score-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>1. Overall Compliance</h2>\n'
            f'<div class="big-score">{score:.0f}%</div>\n'
            f"<div style='text-align:center'>{card_html}</div>\n</div>"
        )

    def _html_standard_grid(self, data: ProfessionalDashboardInput) -> str:
        if not data.compliance_by_standard:
            return (
                '<div class="section"><h2>2. Standards</h2>'
                "<p>No data.</p></div>"
            )
        rows = []
        for std_id in sorted(data.compliance_by_standard.keys()):
            sc = data.compliance_by_standard[std_id]
            name = sc.standard_name or std_id
            rows.append(
                f"<tr><td>{sc.standard_id}</td><td>{name}</td>"
                f"<td>{_fmt_pct(sc.coverage_pct)}</td>"
                f"<td>{_fmt_pct(sc.rule_pass_pct)}</td>"
                f"<td>{_fmt_pct(sc.data_quality)}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. ESRS Standard Compliance</h2>\n'
            "<table><thead><tr><th>Standard</th><th>Name</th><th>Coverage</th>"
            f"<th>Rule Pass</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_quality_gates(self, data: ProfessionalDashboardInput) -> str:
        if not data.quality_gate_status:
            return (
                '<div class="section"><h2>3. Quality Gates</h2>'
                "<p>None configured.</p></div>"
            )
        rows = []
        for g in data.quality_gate_status:
            css = "gate-pass" if g.passed else "gate-fail"
            result = "PASS" if g.passed else "FAIL"
            last = g.last_evaluated.isoformat() if g.last_evaluated else "N/A"
            rows.append(
                f"<tr><td>{g.gate_name}</td><td>{g.score:.1f}%</td>"
                f"<td>{g.threshold:.1f}%</td>"
                f'<td class="{css}">{result}</td><td>{last}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>3. Quality Gate Status</h2>\n'
            "<table><thead><tr><th>Gate</th><th>Score</th><th>Threshold</th>"
            f"<th>Result</th><th>Evaluated</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_approval_pipeline(self, data: ProfessionalDashboardInput) -> str:
        ap = data.approval_status
        cards = [
            (ap.current_level.value, "Current Level"),
            (str(ap.pending_count), "Pending"),
            (f"{ap.avg_approval_hours:.1f}h", "Avg Time"),
            (str(ap.total_approved), "Approved"),
            (str(ap.total_rejected), "Rejected"),
        ]
        card_html = "\n".join(
            f'<div class="score-card"><div class="score-value">{v}</div>'
            f'<div class="score-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>4. Approval Pipeline</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_regulatory_alerts(self, data: ProfessionalDashboardInput) -> str:
        if not data.regulatory_alerts:
            return (
                '<div class="section"><h2>5. Alerts</h2>'
                "<p>No active alerts.</p></div>"
            )
        sorted_alerts = sorted(
            data.regulatory_alerts, key=lambda a: _severity_sort(a.severity)
        )
        rows = []
        for a in sorted_alerts:
            css = f"severity-{a.severity.value.lower()}"
            stds = ", ".join(a.affected_standards) if a.affected_standards else "-"
            action = a.action_required or "-"
            rows.append(
                f'<tr><td class="{css}">{a.severity.value}</td>'
                f"<td>{a.title}</td><td>{stds}</td>"
                f"<td>{a.alert_date.isoformat()}</td><td>{action}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>5. Regulatory Alerts</h2>\n'
            "<table><thead><tr><th>Severity</th><th>Title</th><th>Standards</th>"
            f"<th>Date</th><th>Action</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_benchmark_position(self, data: ProfessionalDashboardInput) -> str:
        if not data.benchmark_position:
            return (
                '<div class="section"><h2>6. Benchmark</h2>'
                "<p>No data.</p></div>"
            )
        bp = data.benchmark_position
        sector = bp.sector or "N/A"
        return (
            '<div class="section">\n<h2>6. Benchmark Position</h2>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{bp.overall_percentile:.0f}th</div>'
            f'<div class="score-label">Overall Percentile</div></div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{bp.trend_direction.value}</div>'
            f'<div class="score-label">Trend</div></div>\n'
            f'<div class="score-card"><div class="score-value">'
            f'{bp.peer_count}</div>'
            f'<div class="score-label">Peers</div></div>\n'
            f"<p><strong>Sector:</strong> {sector}</p>\n</div>"
        )

    def _html_entity_status(self, data: ProfessionalDashboardInput) -> str:
        if not data.entity_status:
            return (
                '<div class="section"><h2>7. Entities</h2>'
                "<p>No entity data.</p></div>"
            )
        rows = []
        for e in data.entity_status:
            rows.append(
                f"<tr><td>{e.entity_name}</td>"
                f"<td>{_fmt_pct(e.data_completeness)}</td>"
                f"<td>{e.calculation_status.value}</td>"
                f"<td>{e.approval_status}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>7. Entity Status Matrix</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Completeness</th>"
            f"<th>Calculation</th><th>Approval</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_slo_adherence(self, data: ProfessionalDashboardInput) -> str:
        if not data.slo_adherence:
            return (
                '<div class="section"><h2>8. SLOs</h2>'
                "<p>None configured.</p></div>"
            )
        rows = []
        for s in data.slo_adherence:
            css = f"slo-{s.status.value.lower().replace('_', '-')}"
            rows.append(
                f"<tr><td>{s.slo_name}</td><td>{_fmt_pct(s.target)}</td>"
                f"<td>{_fmt_pct(s.current)}</td>"
                f'<td class="{css}">{s.status.value}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>8. SLO Adherence</h2>\n'
            "<table><thead><tr><th>SLO</th><th>Target</th><th>Current</th>"
            f"<th>Status</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_deadlines(self, data: ProfessionalDashboardInput) -> str:
        if not data.upcoming_deadlines:
            return (
                '<div class="section"><h2>9. Deadlines</h2>'
                "<p>No deadlines.</p></div>"
            )
        rows = []
        for d in sorted(data.upcoming_deadlines, key=lambda x: x.deadline_date):
            css = f"deadline-{d.status.value.lower().replace('_', '-')}"
            owner = d.owner or "TBD"
            rows.append(
                f"<tr><td>{d.deadline_name}</td><td>{d.deadline_date.isoformat()}</td>"
                f"<td>{d.days_remaining}</td>"
                f'<td class="{css}">{d.status.value}</td>'
                f"<td>{owner}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>9. Deadline Tracker</h2>\n'
            "<table><thead><tr><th>Deadline</th><th>Date</th><th>Days</th>"
            f"<th>Status</th><th>Owner</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: ProfessionalDashboardInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
