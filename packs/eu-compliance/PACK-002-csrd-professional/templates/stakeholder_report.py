# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Stakeholder Engagement Report Template
==========================================================

Stakeholder engagement documentation template covering stakeholder
overview, salience analysis, engagement activities, materiality
influence, participation metrics, and evidence summaries. Aligned
with ESRS 1 stakeholder engagement requirements.

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

class StakeholderCategory(str, Enum):
    """Stakeholder category classification."""
    EMPLOYEES = "EMPLOYEES"
    CUSTOMERS = "CUSTOMERS"
    SUPPLIERS = "SUPPLIERS"
    INVESTORS = "INVESTORS"
    COMMUNITIES = "COMMUNITIES"
    REGULATORS = "REGULATORS"
    NGO = "NGO"
    INDUSTRY_BODIES = "INDUSTRY_BODIES"
    OTHER = "OTHER"


class EngagementType(str, Enum):
    """Type of engagement activity."""
    SURVEY = "SURVEY"
    INTERVIEW = "INTERVIEW"
    WORKSHOP = "WORKSHOP"
    FOCUS_GROUP = "FOCUS_GROUP"
    PUBLIC_CONSULTATION = "PUBLIC_CONSULTATION"
    ADVISORY_PANEL = "ADVISORY_PANEL"
    OTHER = "OTHER"


class SalienceClass(str, Enum):
    """Stakeholder salience classification."""
    DEFINITIVE = "DEFINITIVE"
    DOMINANT = "DOMINANT"
    DEPENDENT = "DEPENDENT"
    DISCRETIONARY = "DISCRETIONARY"
    DEMANDING = "DEMANDING"
    DANGEROUS = "DANGEROUS"
    DORMANT = "DORMANT"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StakeholderSummary(BaseModel):
    """Summary of a stakeholder group."""
    category: StakeholderCategory = Field(..., description="Stakeholder category")
    count: int = Field(0, ge=0, description="Number of stakeholders engaged")
    avg_salience: float = Field(
        0.0, ge=0.0, le=10.0, description="Average salience score"
    )
    engagement_method: str = Field(
        "", description="Primary engagement method used"
    )
    key_concerns: List[str] = Field(
        default_factory=list, description="Top concerns raised"
    )


class SalienceAnalysis(BaseModel):
    """Stakeholder salience analysis results."""
    definitive: List[str] = Field(
        default_factory=list, description="Definitive stakeholders (power+legitimacy+urgency)"
    )
    dominant: List[str] = Field(
        default_factory=list, description="Dominant stakeholders (power+legitimacy)"
    )
    dependent: List[str] = Field(
        default_factory=list, description="Dependent stakeholders (legitimacy+urgency)"
    )
    discretionary: List[str] = Field(
        default_factory=list, description="Discretionary stakeholders (legitimacy only)"
    )


class EngagementActivitySummary(BaseModel):
    """Summary of a single engagement activity."""
    activity_type: EngagementType = Field(..., description="Activity type")
    activity_date: date = Field(..., description="Activity date")
    participants: int = Field(0, ge=0, description="Number of participants")
    stakeholder_groups: List[str] = Field(
        default_factory=list, description="Stakeholder groups involved"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings from this activity"
    )
    location: Optional[str] = Field(None, description="Activity location")


class MaterialityInfluence(BaseModel):
    """Stakeholder influence on materiality assessment."""
    topic: str = Field(..., description="Material topic name")
    stakeholder_avg_impact: float = Field(
        ..., ge=0.0, le=10.0, description="Average impact rating from stakeholders"
    )
    stakeholder_avg_financial: float = Field(
        ..., ge=0.0, le=10.0, description="Average financial rating from stakeholders"
    )
    weighted_importance: float = Field(
        ..., ge=0.0, le=10.0, description="Weighted importance score"
    )
    top_stakeholder_group: Optional[str] = Field(
        None, description="Stakeholder group most concerned"
    )


class ParticipationMetrics(BaseModel):
    """Participation metrics for engagement activities."""
    surveys_sent: int = Field(0, ge=0, description="Surveys distributed")
    surveys_returned: int = Field(0, ge=0, description="Surveys returned")
    response_rate: float = Field(
        0.0, ge=0.0, le=100.0, description="Survey response rate %"
    )
    interviews_conducted: int = Field(0, ge=0, description="Interviews conducted")
    workshops_held: int = Field(0, ge=0, description="Workshops held")
    total_participants: Optional[int] = Field(
        None, ge=0, description="Total unique participants"
    )


class EvidenceSummary(BaseModel):
    """Evidence documentation summary."""
    documents_collected: int = Field(0, ge=0, description="Documents collected")
    survey_records: int = Field(0, ge=0, description="Survey records")
    interview_transcripts: int = Field(0, ge=0, description="Interview transcripts")
    workshop_minutes: int = Field(0, ge=0, description="Workshop minutes")
    total_evidence_items: Optional[int] = Field(
        None, ge=0, description="Total evidence items"
    )


class StakeholderReportInput(BaseModel):
    """Complete input for the stakeholder engagement report."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    stakeholders: List[StakeholderSummary] = Field(
        default_factory=list, description="Stakeholder summaries"
    )
    salience_analysis: SalienceAnalysis = Field(
        default_factory=SalienceAnalysis, description="Salience analysis"
    )
    engagement_activities: List[EngagementActivitySummary] = Field(
        default_factory=list, description="Engagement activities"
    )
    materiality_influence: List[MaterialityInfluence] = Field(
        default_factory=list, description="Materiality influence data"
    )
    participation_metrics: ParticipationMetrics = Field(
        default_factory=ParticipationMetrics, description="Participation metrics"
    )
    evidence_summary: EvidenceSummary = Field(
        default_factory=EvidenceSummary, description="Evidence documentation"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format numeric value."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class StakeholderReportTemplate:
    """Generate stakeholder engagement documentation report.

    Sections:
        1. Stakeholder Overview
        2. Salience Map
        3. Engagement Activity Log
        4. Materiality Influence Analysis
        5. Participation Metrics
        6. Evidence Summary
        7. ESRS 1 Compliance Statement

    Example:
        >>> template = StakeholderReportTemplate()
        >>> data = StakeholderReportInput(
        ...     organization_name="Acme", reporting_year=2025
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "stakeholder_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the stakeholder report template."""
        self._render_timestamp: Optional[datetime] = None

    def render_markdown(self, data: StakeholderReportInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_stakeholder_overview(data),
            self._md_salience_map(data),
            self._md_engagement_log(data),
            self._md_materiality_influence(data),
            self._md_participation_metrics(data),
            self._md_evidence_summary(data),
            self._md_esrs_compliance(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: StakeholderReportInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_stakeholder_overview(data),
            self._html_salience_map(data),
            self._html_engagement_log(data),
            self._html_materiality_influence(data),
            self._html_participation_metrics(data),
            self._html_evidence_summary(data),
            self._html_esrs_compliance(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: StakeholderReportInput) -> Dict[str, Any]:
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
            "stakeholders": [s.model_dump(mode="json") for s in data.stakeholders],
            "salience_analysis": data.salience_analysis.model_dump(mode="json"),
            "engagement_activities": [
                a.model_dump(mode="json") for a in data.engagement_activities
            ],
            "materiality_influence": [
                m.model_dump(mode="json") for m in data.materiality_influence
            ],
            "participation_metrics": data.participation_metrics.model_dump(mode="json"),
            "evidence_summary": data.evidence_summary.model_dump(mode="json"),
        }

    def _compute_provenance(self, data: StakeholderReportInput) -> str:
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: StakeholderReportInput) -> str:
        total_groups = len(data.stakeholders)
        total_activities = len(data.engagement_activities)
        return (
            f"# Stakeholder Engagement Report - {data.organization_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Groups:** {total_groups} | "
            f"**Activities:** {total_activities}\n\n---"
        )

    def _md_stakeholder_overview(self, data: StakeholderReportInput) -> str:
        if not data.stakeholders:
            return "## 1. Stakeholder Overview\n\nNo stakeholder data available."
        lines = [
            "## 1. Stakeholder Overview",
            "",
            "| Category | Engaged | Avg Salience | Method | Key Concerns |",
            "|----------|---------|-------------|--------|-------------|",
        ]
        for s in data.stakeholders:
            concerns = "; ".join(s.key_concerns[:3]) if s.key_concerns else "-"
            method = s.engagement_method or "-"
            lines.append(
                f"| {s.category.value} | {s.count} | {s.avg_salience:.1f}/10 "
                f"| {method} | {concerns} |"
            )
        return "\n".join(lines)

    def _md_salience_map(self, data: StakeholderReportInput) -> str:
        sa = data.salience_analysis
        lines = ["## 2. Salience Map", ""]
        categories = [
            ("Definitive (Power + Legitimacy + Urgency)", sa.definitive),
            ("Dominant (Power + Legitimacy)", sa.dominant),
            ("Dependent (Legitimacy + Urgency)", sa.dependent),
            ("Discretionary (Legitimacy)", sa.discretionary),
        ]
        for label, groups in categories:
            group_text = ", ".join(groups) if groups else "None identified"
            lines.append(f"- **{label}:** {group_text}")
        return "\n".join(lines)

    def _md_engagement_log(self, data: StakeholderReportInput) -> str:
        if not data.engagement_activities:
            return "## 3. Engagement Activity Log\n\nNo activities recorded."
        lines = [
            "## 3. Engagement Activity Log",
            "",
            "| Type | Date | Participants | Groups | Key Findings |",
            "|------|------|-------------|--------|-------------|",
        ]
        for a in sorted(data.engagement_activities, key=lambda x: x.activity_date, reverse=True):
            groups = ", ".join(a.stakeholder_groups) if a.stakeholder_groups else "-"
            findings = "; ".join(a.key_findings[:2]) if a.key_findings else "-"
            lines.append(
                f"| {a.activity_type.value} | {a.activity_date.isoformat()} "
                f"| {a.participants} | {groups} | {findings} |"
            )
        return "\n".join(lines)

    def _md_materiality_influence(self, data: StakeholderReportInput) -> str:
        if not data.materiality_influence:
            return "## 4. Materiality Influence\n\nNo materiality influence data."
        lines = [
            "## 4. Materiality Influence Analysis",
            "",
            "| Topic | Impact Rating | Financial Rating | Weighted Score | Top Group |",
            "|-------|-------------|-----------------|----------------|-----------|",
        ]
        for m in sorted(data.materiality_influence, key=lambda x: x.weighted_importance, reverse=True):
            group = m.top_stakeholder_group or "-"
            lines.append(
                f"| {m.topic} | {m.stakeholder_avg_impact:.1f}/10 "
                f"| {m.stakeholder_avg_financial:.1f}/10 "
                f"| {m.weighted_importance:.1f}/10 | {group} |"
            )
        return "\n".join(lines)

    def _md_participation_metrics(self, data: StakeholderReportInput) -> str:
        pm = data.participation_metrics
        total = pm.total_participants if pm.total_participants is not None else "N/A"
        return (
            "## 5. Participation Metrics\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Surveys Sent | {pm.surveys_sent} |\n"
            f"| Surveys Returned | {pm.surveys_returned} |\n"
            f"| Response Rate | {_fmt_pct(pm.response_rate)} |\n"
            f"| Interviews Conducted | {pm.interviews_conducted} |\n"
            f"| Workshops Held | {pm.workshops_held} |\n"
            f"| Total Unique Participants | {total} |"
        )

    def _md_evidence_summary(self, data: StakeholderReportInput) -> str:
        es = data.evidence_summary
        total = es.total_evidence_items if es.total_evidence_items is not None else "N/A"
        return (
            "## 6. Evidence Summary\n\n"
            "| Evidence Type | Count |\n"
            "|--------------|-------|\n"
            f"| Documents Collected | {es.documents_collected} |\n"
            f"| Survey Records | {es.survey_records} |\n"
            f"| Interview Transcripts | {es.interview_transcripts} |\n"
            f"| Workshop Minutes | {es.workshop_minutes} |\n"
            f"| **Total Evidence Items** | **{total}** |"
        )

    def _md_esrs_compliance(self, data: StakeholderReportInput) -> str:
        has_engagement = len(data.engagement_activities) > 0
        has_salience = bool(data.salience_analysis.definitive or data.salience_analysis.dominant)
        has_materiality = len(data.materiality_influence) > 0
        has_evidence = (data.evidence_summary.documents_collected > 0
                        or data.evidence_summary.survey_records > 0)
        checks = [
            ("Stakeholder identification and mapping", len(data.stakeholders) > 0),
            ("Salience analysis completed", has_salience),
            ("Engagement activities documented", has_engagement),
            ("Materiality influence analysis", has_materiality),
            ("Evidence and documentation", has_evidence),
        ]
        lines = ["## 7. ESRS 1 Compliance Statement", ""]
        for label, done in checks:
            mark = "[x]" if done else "[ ]"
            lines.append(f"- {mark} {label}")
        compliant = all(done for _, done in checks)
        lines.extend([
            "",
            f"**ESRS 1 Stakeholder Engagement Compliance:** "
            f"{'COMPLIANT' if compliant else 'GAPS IDENTIFIED'}",
        ])
        return "\n".join(lines)

    def _md_footer(self, data: StakeholderReportInput) -> str:
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
            f"<title>Stakeholder Report - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".salience-box{background:#f8f9fa;border-left:4px solid #533483;"
            "padding:0.75rem;margin:0.5rem 0;border-radius:0 6px 6px 0;}\n"
            ".check-pass{color:#1a7f37;}\n"
            ".check-fail{color:#cf222e;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: StakeholderReportInput) -> str:
        return (
            '<div class="section">\n'
            f"<h1>Stakeholder Engagement Report &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Year:</strong> {data.reporting_year} | "
            f"<strong>Groups:</strong> {len(data.stakeholders)} | "
            f"<strong>Activities:</strong> {len(data.engagement_activities)}</p>\n"
            "<hr>\n</div>"
        )

    def _html_stakeholder_overview(self, data: StakeholderReportInput) -> str:
        if not data.stakeholders:
            return '<div class="section"><h2>1. Overview</h2><p>No data.</p></div>'
        rows = []
        for s in data.stakeholders:
            concerns = "; ".join(s.key_concerns[:3]) if s.key_concerns else "-"
            method = s.engagement_method or "-"
            rows.append(
                f"<tr><td>{s.category.value}</td><td>{s.count}</td>"
                f"<td>{s.avg_salience:.1f}/10</td><td>{method}</td>"
                f"<td>{concerns}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>1. Stakeholder Overview</h2>\n'
            "<table><thead><tr><th>Category</th><th>Engaged</th>"
            "<th>Salience</th><th>Method</th><th>Concerns</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_salience_map(self, data: StakeholderReportInput) -> str:
        sa = data.salience_analysis
        categories = [
            ("Definitive", sa.definitive),
            ("Dominant", sa.dominant),
            ("Dependent", sa.dependent),
            ("Discretionary", sa.discretionary),
        ]
        parts = ['<div class="section">\n<h2>2. Salience Map</h2>\n']
        for label, groups in categories:
            text = ", ".join(groups) if groups else "None identified"
            parts.append(
                f'<div class="salience-box"><strong>{label}:</strong> {text}</div>\n'
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_engagement_log(self, data: StakeholderReportInput) -> str:
        if not data.engagement_activities:
            return '<div class="section"><h2>3. Activities</h2><p>None.</p></div>'
        rows = []
        for a in sorted(data.engagement_activities, key=lambda x: x.activity_date, reverse=True):
            groups = ", ".join(a.stakeholder_groups) if a.stakeholder_groups else "-"
            findings = "; ".join(a.key_findings[:2]) if a.key_findings else "-"
            rows.append(
                f"<tr><td>{a.activity_type.value}</td><td>{a.activity_date.isoformat()}</td>"
                f"<td>{a.participants}</td><td>{groups}</td><td>{findings}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>3. Engagement Activity Log</h2>\n'
            "<table><thead><tr><th>Type</th><th>Date</th><th>Participants</th>"
            f"<th>Groups</th><th>Findings</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_materiality_influence(self, data: StakeholderReportInput) -> str:
        if not data.materiality_influence:
            return '<div class="section"><h2>4. Influence</h2><p>No data.</p></div>'
        rows = []
        for m in sorted(data.materiality_influence, key=lambda x: x.weighted_importance, reverse=True):
            group = m.top_stakeholder_group or "-"
            rows.append(
                f"<tr><td>{m.topic}</td><td>{m.stakeholder_avg_impact:.1f}/10</td>"
                f"<td>{m.stakeholder_avg_financial:.1f}/10</td>"
                f"<td>{m.weighted_importance:.1f}/10</td><td>{group}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Materiality Influence Analysis</h2>\n'
            "<table><thead><tr><th>Topic</th><th>Impact</th><th>Financial</th>"
            f"<th>Weighted</th><th>Top Group</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_participation_metrics(self, data: StakeholderReportInput) -> str:
        pm = data.participation_metrics
        cards = [
            (str(pm.surveys_sent), "Surveys Sent"),
            (str(pm.surveys_returned), "Returned"),
            (_fmt_pct(pm.response_rate), "Response Rate"),
            (str(pm.interviews_conducted), "Interviews"),
            (str(pm.workshops_held), "Workshops"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>5. Participation Metrics</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_evidence_summary(self, data: StakeholderReportInput) -> str:
        es = data.evidence_summary
        total = es.total_evidence_items if es.total_evidence_items is not None else "N/A"
        rows = [
            f"<tr><td>Documents Collected</td><td>{es.documents_collected}</td></tr>",
            f"<tr><td>Survey Records</td><td>{es.survey_records}</td></tr>",
            f"<tr><td>Interview Transcripts</td><td>{es.interview_transcripts}</td></tr>",
            f"<tr><td>Workshop Minutes</td><td>{es.workshop_minutes}</td></tr>",
            f"<tr style='font-weight:bold'><td>Total</td><td>{total}</td></tr>",
        ]
        return (
            '<div class="section">\n<h2>6. Evidence Summary</h2>\n'
            "<table><thead><tr><th>Type</th><th>Count</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_esrs_compliance(self, data: StakeholderReportInput) -> str:
        has_engagement = len(data.engagement_activities) > 0
        has_salience = bool(data.salience_analysis.definitive or data.salience_analysis.dominant)
        has_materiality = len(data.materiality_influence) > 0
        has_evidence = (data.evidence_summary.documents_collected > 0
                        or data.evidence_summary.survey_records > 0)
        checks = [
            ("Stakeholder identification and mapping", len(data.stakeholders) > 0),
            ("Salience analysis completed", has_salience),
            ("Engagement activities documented", has_engagement),
            ("Materiality influence analysis", has_materiality),
            ("Evidence and documentation", has_evidence),
        ]
        items = []
        for label, done in checks:
            css = "check-pass" if done else "check-fail"
            mark = "&#10003;" if done else "&#10007;"
            items.append(f'<li class="{css}">{mark} {label}</li>')
        compliant = all(done for _, done in checks)
        result_css = "check-pass" if compliant else "check-fail"
        result_text = "COMPLIANT" if compliant else "GAPS IDENTIFIED"
        return (
            '<div class="section">\n<h2>7. ESRS 1 Compliance Statement</h2>\n'
            f'<ul style="list-style:none;padding:0">{"".join(items)}</ul>\n'
            f'<p><strong>Status:</strong> <span class="{result_css}">'
            f'{result_text}</span></p>\n</div>'
        )

    def _html_footer(self, data: StakeholderReportInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
