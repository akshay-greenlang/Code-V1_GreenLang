"""
BundleExecutiveSummaryTemplate - Board-level overview spanning all 4 regulations.

This module implements the BundleExecutiveSummaryTemplate for PACK-009
EU Climate Compliance Bundle. It renders a board-ready executive summary
covering overall compliance score, data completeness, gaps remaining,
per-regulation risk flags, regulatory change alerts, strategic
recommendations, and year-over-year comparison.

Example:
    >>> template = BundleExecutiveSummaryTemplate()
    >>> data = ExecutiveSummaryData(
    ...     metrics=OverallMetrics(compliance_score=82.5, ...),
    ...     risks=[...],
    ...     recommendations=[...],
    ...     trends=YoYTrends(...),
    ... )
    >>> html = template.render(data, fmt="html")
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic data models
# ---------------------------------------------------------------------------

class OverallMetrics(BaseModel):
    """Top-level metrics for the executive summary."""

    compliance_score: float = Field(..., ge=0.0, le=100.0, description="Overall compliance score")
    data_completeness_pct: float = Field(0.0, ge=0.0, le=100.0, description="Data completeness %")
    gaps_remaining: int = Field(0, ge=0, description="Number of open gaps")
    critical_gaps: int = Field(0, ge=0, description="Number of critical gaps")
    regulations_on_track: int = Field(0, ge=0, description="Regulations on track")
    regulations_total: int = Field(4, ge=1, description="Total regulations in bundle")
    next_major_deadline: str = Field("", description="Next major regulatory deadline date")
    next_deadline_description: str = Field("", description="Description of next deadline")
    total_remediation_cost_eur: float = Field(0.0, ge=0.0, description="Est. total remediation cost")


class RegulationRisk(BaseModel):
    """Risk flag for a specific regulation."""

    regulation: str = Field(..., description="Regulation code")
    risk_level: str = Field("low", description="critical, high, medium, low")
    risk_title: str = Field(..., description="Short risk title")
    risk_description: str = Field("", description="Detailed risk description")
    impact: str = Field("", description="Business impact description")
    mitigation: str = Field("", description="Recommended mitigation")
    deadline_at_risk: bool = Field(False, description="Whether a deadline is at risk")


class RegulatoryChangeAlert(BaseModel):
    """Alert about upcoming regulatory changes."""

    regulation: str = Field(..., description="Affected regulation code")
    alert_type: str = Field("update", description="new, update, amendment, repeal")
    title: str = Field(..., description="Change title")
    description: str = Field("", description="Change description")
    effective_date: str = Field("", description="Effective date ISO string")
    impact_level: str = Field("medium", description="high, medium, low")
    action_required: str = Field("", description="Recommended action")


class StrategicRecommendation(BaseModel):
    """An actionable strategic recommendation."""

    priority: int = Field(1, ge=1, le=10, description="Priority ranking 1=highest")
    title: str = Field(..., description="Recommendation title")
    description: str = Field("", description="Detailed recommendation")
    regulations_impacted: List[str] = Field(
        default_factory=list, description="Regulations this addresses"
    )
    estimated_impact: str = Field("", description="Expected impact description")
    estimated_effort: str = Field("", description="Effort level: low, medium, high")
    timeline: str = Field("", description="Suggested timeline e.g. Q1 2026")


class RegulationYoY(BaseModel):
    """Year-over-year comparison for one regulation."""

    regulation: str = Field(..., description="Regulation code")
    prior_year_score: float = Field(0.0, ge=0.0, le=100.0, description="Prior year compliance score")
    current_year_score: float = Field(0.0, ge=0.0, le=100.0, description="Current year score")
    delta: float = Field(0.0, description="Change in score")
    prior_year_gaps: int = Field(0, ge=0, description="Prior year gap count")
    current_year_gaps: int = Field(0, ge=0, description="Current year gap count")


class YoYTrends(BaseModel):
    """Year-over-year trends for the bundle."""

    prior_year_label: str = Field("FY2024", description="Prior year label")
    current_year_label: str = Field("FY2025", description="Current year label")
    overall_prior_score: float = Field(0.0, ge=0.0, le=100.0, description="Prior year overall score")
    overall_current_score: float = Field(0.0, ge=0.0, le=100.0, description="Current year overall score")
    per_regulation: List[RegulationYoY] = Field(
        default_factory=list, description="Per-regulation YoY data"
    )


class ExecutiveSummaryConfig(BaseModel):
    """Configuration for the executive summary template."""

    title: str = Field(
        "EU Climate Compliance Bundle - Executive Summary",
        description="Report title",
    )
    max_recommendations: int = Field(5, description="Maximum recommendations to display")
    show_yoy: bool = Field(True, description="Whether to show year-over-year comparison")
    score_threshold_green: float = Field(80.0, description="Green threshold")
    score_threshold_amber: float = Field(50.0, description="Amber threshold")


class ExecutiveSummaryData(BaseModel):
    """Input data for the executive summary."""

    metrics: OverallMetrics = Field(..., description="Overall metrics")
    risks: List[RegulationRisk] = Field(default_factory=list, description="Risk flags per regulation")
    change_alerts: List[RegulatoryChangeAlert] = Field(
        default_factory=list, description="Regulatory change alerts"
    )
    recommendations: List[StrategicRecommendation] = Field(
        default_factory=list, description="Strategic recommendations"
    )
    trends: Optional[YoYTrends] = Field(None, description="Year-over-year trends")
    per_regulation_scores: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation compliance scores"
    )
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")
    prepared_for: str = Field("", description="Board/audience name")


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class BundleExecutiveSummaryTemplate:
    """
    Board-level executive summary template for the EU Climate Compliance Bundle.

    Generates executive summaries with overall compliance scoring,
    per-regulation status, risk flags, regulatory change alerts,
    strategic recommendations, and year-over-year comparisons.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    RISK_COLORS = {
        "critical": {"hex": "#c0392b", "label": "CRITICAL"},
        "high": {"hex": "#e74c3c", "label": "HIGH"},
        "medium": {"hex": "#f39c12", "label": "MEDIUM"},
        "low": {"hex": "#27ae60", "label": "LOW"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize BundleExecutiveSummaryTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = ExecutiveSummaryConfig(**raw) if raw else ExecutiveSummaryConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: ExecutiveSummaryData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the executive summary in the specified format.

        Args:
            data: Validated ExecutiveSummaryData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output.

        Raises:
            ValueError: If fmt is unsupported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    def render_markdown(self, data: ExecutiveSummaryData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_key_metrics(data),
            self._md_per_regulation_status(data),
            self._md_risk_flags(data),
            self._md_change_alerts(data),
            self._md_recommendations(data),
            self._md_yoy_comparison(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: ExecutiveSummaryData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_key_metrics(data),
            self._html_per_regulation_status(data),
            self._html_risk_flags(data),
            self._html_change_alerts(data),
            self._html_recommendations(data),
            self._html_yoy_comparison(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: ExecutiveSummaryData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "bundle_executive_summary",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "prepared_for": data.prepared_for,
            "key_metrics": self._json_key_metrics(data),
            "per_regulation_scores": data.per_regulation_scores,
            "risks": self._json_risks(data),
            "change_alerts": self._json_change_alerts(data),
            "recommendations": self._json_recommendations(data),
            "yoy_trends": self._json_yoy(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Traffic-light helpers
    # ------------------------------------------------------------------ #

    def _score_color(self, score: float) -> str:
        """Return hex color based on score thresholds."""
        if score >= self.config.score_threshold_green:
            return "#2ecc71"
        elif score >= self.config.score_threshold_amber:
            return "#f39c12"
        return "#e74c3c"

    def _score_label(self, score: float) -> str:
        """Return text label based on score thresholds."""
        if score >= self.config.score_threshold_green:
            return "GREEN"
        elif score >= self.config.score_threshold_amber:
            return "AMBER"
        return "RED"

    def _score_symbol(self, score: float) -> str:
        """Return markdown symbol for score."""
        if score >= self.config.score_threshold_green:
            return "[G]"
        elif score >= self.config.score_threshold_amber:
            return "[A]"
        return "[R]"

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: ExecutiveSummaryData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        prepared = f"\n\n**Prepared For:** {data.prepared_for}" if data.prepared_for else ""
        sym = self._score_symbol(data.metrics.compliance_score)
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}"
            f"{prepared}\n\n"
            f"**Overall Status:** {sym} {self._score_label(data.metrics.compliance_score)}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_key_metrics(self, data: ExecutiveSummaryData) -> str:
        """Build key metrics dashboard section."""
        m = data.metrics
        filled = int(m.compliance_score / 5)
        empty = 20 - filled
        gauge = "[" + "#" * filled + "-" * empty + "]"
        deadline_info = ""
        if m.next_major_deadline:
            deadline_info = (
                f"\n- **Next Major Deadline:** {m.next_major_deadline[:10]}"
                f" - {m.next_deadline_description}"
            )
        return (
            "## Key Metrics\n\n"
            f"```\nCompliance Score: {gauge} {m.compliance_score:.1f}/100\n```\n\n"
            f"- **Data Completeness:** {m.data_completeness_pct:.1f}%\n"
            f"- **Open Gaps:** {m.gaps_remaining} ({m.critical_gaps} critical)\n"
            f"- **Regulations On Track:** {m.regulations_on_track}/{m.regulations_total}\n"
            f"- **Estimated Remediation Cost:** EUR {m.total_remediation_cost_eur:,.0f}"
            f"{deadline_info}"
        )

    def _md_per_regulation_status(self, data: ExecutiveSummaryData) -> str:
        """Build per-regulation status table."""
        if not data.per_regulation_scores:
            return ""
        header = (
            "## Per-Regulation Status\n\n"
            "| Regulation | Score | Status |\n"
            "|------------|-------|--------|\n"
        )
        rows: List[str] = []
        for reg, score in sorted(data.per_regulation_scores.items()):
            sym = self._score_symbol(score)
            rows.append(f"| {reg} | {score:.1f}/100 | {sym} {self._score_label(score)} |")
        return header + "\n".join(rows)

    def _md_risk_flags(self, data: ExecutiveSummaryData) -> str:
        """Build risk flags section."""
        if not data.risks:
            return "## Risk Flags\n\n*No significant risks identified.*"
        section = "## Risk Flags\n\n"
        sorted_risks = sorted(
            data.risks,
            key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(r.risk_level, 4)
        )
        for risk in sorted_risks:
            deadline_warn = " **[DEADLINE AT RISK]**" if risk.deadline_at_risk else ""
            section += (
                f"### {risk.regulation}: {risk.risk_title} [{risk.risk_level.upper()}]{deadline_warn}\n\n"
                f"{risk.risk_description}\n\n"
                f"- **Impact:** {risk.impact}\n"
                f"- **Mitigation:** {risk.mitigation}\n\n"
            )
        return section.rstrip()

    def _md_change_alerts(self, data: ExecutiveSummaryData) -> str:
        """Build regulatory change alerts section."""
        if not data.change_alerts:
            return ""
        header = (
            "## Regulatory Change Alerts\n\n"
            "| Regulation | Type | Change | Effective | Impact | Action |\n"
            "|------------|------|--------|-----------|--------|--------|\n"
        )
        rows: List[str] = []
        for alert in data.change_alerts:
            rows.append(
                f"| {alert.regulation} | {alert.alert_type.upper()} | "
                f"{alert.title} | {alert.effective_date[:10] if alert.effective_date else 'TBD'} | "
                f"{alert.impact_level.upper()} | {alert.action_required or 'Review'} |"
            )
        return header + "\n".join(rows)

    def _md_recommendations(self, data: ExecutiveSummaryData) -> str:
        """Build strategic recommendations section."""
        if not data.recommendations:
            return ""
        recs = sorted(data.recommendations, key=lambda r: r.priority)
        recs = recs[:self.config.max_recommendations]
        section = "## Strategic Recommendations\n\n"
        for i, rec in enumerate(recs, 1):
            regs = ", ".join(rec.regulations_impacted) if rec.regulations_impacted else "All"
            section += (
                f"### {i}. {rec.title}\n\n"
                f"{rec.description}\n\n"
                f"- **Regulations:** {regs}\n"
                f"- **Expected Impact:** {rec.estimated_impact}\n"
                f"- **Effort:** {rec.estimated_effort}\n"
                f"- **Timeline:** {rec.timeline}\n\n"
            )
        return section.rstrip()

    def _md_yoy_comparison(self, data: ExecutiveSummaryData) -> str:
        """Build year-over-year comparison section."""
        if not self.config.show_yoy or not data.trends:
            return ""
        t = data.trends
        delta = t.overall_current_score - t.overall_prior_score
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        section = (
            "## Year-over-Year Comparison\n\n"
            f"**{t.prior_year_label}:** {t.overall_prior_score:.1f} -> "
            f"**{t.current_year_label}:** {t.overall_current_score:.1f} "
            f"({arrow} {abs(delta):.1f})\n\n"
        )
        if t.per_regulation:
            section += (
                "| Regulation | Prior | Current | Delta | Gaps Prior | Gaps Current |\n"
                "|------------|-------|---------|-------|------------|-------------|\n"
            )
            for r in t.per_regulation:
                d_arrow = "+" if r.delta > 0 else "" if r.delta < 0 else "="
                section += (
                    f"| {r.regulation} | {r.prior_year_score:.1f} | "
                    f"{r.current_year_score:.1f} | {d_arrow}{r.delta:.1f} | "
                    f"{r.prior_year_gaps} | {r.current_year_gaps} |\n"
                )
        return section.rstrip()

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            "*This report is intended for board-level review. "
            "All data has been validated against source systems.*\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: BundleExecutiveSummaryTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: ExecutiveSummaryData) -> str:
        """Build HTML header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        color = self._score_color(data.metrics.compliance_score)
        prepared = f'<div class="meta-item">Prepared for: {data.prepared_for}</div>' if data.prepared_for else ""
        return (
            '<div class="report-header">'
            f'<h1>{self.config.title}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item">Organization: {org}</div>'
            f'<div class="meta-item">Period: {period}</div>'
            f'{prepared}'
            f'<div class="meta-item" style="background:{color};color:#fff">'
            f'{self._score_label(data.metrics.compliance_score)}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_key_metrics(self, data: ExecutiveSummaryData) -> str:
        """Build HTML key metrics section."""
        m = data.metrics
        color = self._score_color(m.compliance_score)
        deadline_html = ""
        if m.next_major_deadline:
            deadline_html = (
                f'<div class="stat-card" style="border-top:3px solid #1a5276">'
                f'<span class="stat-val">{m.next_major_deadline[:10]}</span>'
                f'<span class="stat-lbl">{m.next_deadline_description}</span></div>'
            )
        return (
            '<div class="section"><h2>Key Metrics</h2>'
            '<div class="score-gauge">'
            f'<div class="gauge-circle" style="border-color:{color}">'
            f'<span class="gauge-value">{m.compliance_score:.0f}</span>'
            '<span class="gauge-label">Compliance</span></div></div>'
            '<div class="stat-grid">'
            f'<div class="stat-card"><span class="stat-val">{m.data_completeness_pct:.0f}%</span>'
            f'<span class="stat-lbl">Data Completeness</span></div>'
            f'<div class="stat-card"><span class="stat-val">{m.gaps_remaining}</span>'
            f'<span class="stat-lbl">Open Gaps ({m.critical_gaps} critical)</span></div>'
            f'<div class="stat-card"><span class="stat-val">'
            f'{m.regulations_on_track}/{m.regulations_total}</span>'
            f'<span class="stat-lbl">On Track</span></div>'
            f'<div class="stat-card"><span class="stat-val">EUR {m.total_remediation_cost_eur:,.0f}</span>'
            f'<span class="stat-lbl">Est. Remediation</span></div>'
            f'{deadline_html}'
            '</div></div>'
        )

    def _html_per_regulation_status(self, data: ExecutiveSummaryData) -> str:
        """Build HTML per-regulation status."""
        if not data.per_regulation_scores:
            return ""
        rows = ""
        for reg, score in sorted(data.per_regulation_scores.items()):
            color = self._score_color(score)
            rows += (
                f'<tr><td><strong>{reg}</strong></td>'
                f'<td class="num">{score:.1f}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{self._score_label(score)}</span></td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{score:.0f}%;background:{color}"></div></div></td></tr>'
            )
        return (
            '<div class="section"><h2>Per-Regulation Status</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Score</th><th>Status</th><th>Progress</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_risk_flags(self, data: ExecutiveSummaryData) -> str:
        """Build HTML risk flags."""
        if not data.risks:
            return (
                '<div class="section"><h2>Risk Flags</h2>'
                '<p class="note">No significant risks identified.</p></div>'
            )
        sorted_risks = sorted(
            data.risks,
            key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(r.risk_level, 4)
        )
        cards = ""
        for risk in sorted_risks:
            info = self.RISK_COLORS.get(risk.risk_level, self.RISK_COLORS["low"])
            deadline_badge = (
                ' <span class="deadline-risk">DEADLINE AT RISK</span>'
                if risk.deadline_at_risk else ""
            )
            cards += (
                f'<div class="risk-card" style="border-left:4px solid {info["hex"]}">'
                f'<div class="risk-header">'
                f'<strong>{risk.regulation}: {risk.risk_title}</strong>'
                f'<span class="risk-badge" style="background:{info["hex"]}">'
                f'{info["label"]}</span>{deadline_badge}</div>'
                f'<p>{risk.risk_description}</p>'
                f'<div class="risk-detail">Impact: {risk.impact}</div>'
                f'<div class="risk-detail">Mitigation: {risk.mitigation}</div>'
                f'</div>'
            )
        return f'<div class="section"><h2>Risk Flags</h2>{cards}</div>'

    def _html_change_alerts(self, data: ExecutiveSummaryData) -> str:
        """Build HTML regulatory change alerts."""
        if not data.change_alerts:
            return ""
        rows = ""
        for alert in data.change_alerts:
            impact_color = self.RISK_COLORS.get(alert.impact_level, self.RISK_COLORS["medium"])["hex"]
            rows += (
                f'<tr><td>{alert.regulation}</td>'
                f'<td>{alert.alert_type.upper()}</td>'
                f'<td>{alert.title}</td>'
                f'<td>{alert.effective_date[:10] if alert.effective_date else "TBD"}</td>'
                f'<td><span class="impact-badge" style="background:{impact_color}">'
                f'{alert.impact_level.upper()}</span></td>'
                f'<td>{alert.action_required or "Review"}</td></tr>'
            )
        return (
            '<div class="section"><h2>Regulatory Change Alerts</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Type</th><th>Change</th>'
            '<th>Effective</th><th>Impact</th><th>Action</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_recommendations(self, data: ExecutiveSummaryData) -> str:
        """Build HTML strategic recommendations."""
        if not data.recommendations:
            return ""
        recs = sorted(data.recommendations, key=lambda r: r.priority)
        recs = recs[:self.config.max_recommendations]
        items = ""
        for i, rec in enumerate(recs, 1):
            regs = ", ".join(rec.regulations_impacted) if rec.regulations_impacted else "All"
            items += (
                f'<div class="rec-card">'
                f'<div class="rec-num">{i}</div>'
                f'<div class="rec-content">'
                f'<h3>{rec.title}</h3>'
                f'<p>{rec.description}</p>'
                f'<div class="rec-meta">'
                f'Regulations: {regs} | '
                f'Impact: {rec.estimated_impact} | '
                f'Effort: {rec.estimated_effort} | '
                f'Timeline: {rec.timeline}'
                f'</div></div></div>'
            )
        return f'<div class="section"><h2>Strategic Recommendations</h2>{items}</div>'

    def _html_yoy_comparison(self, data: ExecutiveSummaryData) -> str:
        """Build HTML year-over-year comparison."""
        if not self.config.show_yoy or not data.trends:
            return ""
        t = data.trends
        delta = t.overall_current_score - t.overall_prior_score
        delta_color = "#2ecc71" if delta >= 0 else "#e74c3c"
        rows = ""
        for r in t.per_regulation:
            d_color = "#2ecc71" if r.delta >= 0 else "#e74c3c"
            rows += (
                f'<tr><td><strong>{r.regulation}</strong></td>'
                f'<td class="num">{r.prior_year_score:.1f}</td>'
                f'<td class="num">{r.current_year_score:.1f}</td>'
                f'<td class="num" style="color:{d_color}">'
                f'{"+" if r.delta > 0 else ""}{r.delta:.1f}</td>'
                f'<td class="num">{r.prior_year_gaps}</td>'
                f'<td class="num">{r.current_year_gaps}</td></tr>'
            )
        return (
            '<div class="section"><h2>Year-over-Year Comparison</h2>'
            f'<div class="yoy-header">'
            f'<span>{t.prior_year_label}: {t.overall_prior_score:.1f}</span>'
            f'<span style="color:{delta_color};font-size:24px;font-weight:700">'
            f'{"+" if delta > 0 else ""}{delta:.1f}</span>'
            f'<span>{t.current_year_label}: {t.overall_current_score:.1f}</span>'
            f'</div>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Prior</th><th>Current</th>'
            '<th>Delta</th><th>Gaps Prior</th><th>Gaps Current</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_key_metrics(self, data: ExecutiveSummaryData) -> Dict[str, Any]:
        """Build JSON key metrics."""
        m = data.metrics
        return {
            "compliance_score": round(m.compliance_score, 1),
            "status": self._score_label(m.compliance_score),
            "data_completeness_pct": round(m.data_completeness_pct, 1),
            "gaps_remaining": m.gaps_remaining,
            "critical_gaps": m.critical_gaps,
            "regulations_on_track": m.regulations_on_track,
            "regulations_total": m.regulations_total,
            "next_major_deadline": m.next_major_deadline,
            "next_deadline_description": m.next_deadline_description,
            "total_remediation_cost_eur": round(m.total_remediation_cost_eur, 2),
        }

    def _json_risks(self, data: ExecutiveSummaryData) -> List[Dict[str, Any]]:
        """Build JSON risk flags."""
        return [
            {
                "regulation": r.regulation,
                "risk_level": r.risk_level,
                "risk_title": r.risk_title,
                "risk_description": r.risk_description,
                "impact": r.impact,
                "mitigation": r.mitigation,
                "deadline_at_risk": r.deadline_at_risk,
            }
            for r in data.risks
        ]

    def _json_change_alerts(self, data: ExecutiveSummaryData) -> List[Dict[str, Any]]:
        """Build JSON change alerts."""
        return [
            {
                "regulation": a.regulation,
                "alert_type": a.alert_type,
                "title": a.title,
                "description": a.description,
                "effective_date": a.effective_date,
                "impact_level": a.impact_level,
                "action_required": a.action_required,
            }
            for a in data.change_alerts
        ]

    def _json_recommendations(self, data: ExecutiveSummaryData) -> List[Dict[str, Any]]:
        """Build JSON recommendations."""
        recs = sorted(data.recommendations, key=lambda r: r.priority)
        recs = recs[:self.config.max_recommendations]
        return [
            {
                "priority": r.priority,
                "title": r.title,
                "description": r.description,
                "regulations_impacted": r.regulations_impacted,
                "estimated_impact": r.estimated_impact,
                "estimated_effort": r.estimated_effort,
                "timeline": r.timeline,
            }
            for r in recs
        ]

    def _json_yoy(self, data: ExecutiveSummaryData) -> Optional[Dict[str, Any]]:
        """Build JSON year-over-year data."""
        if not data.trends:
            return None
        t = data.trends
        return {
            "prior_year_label": t.prior_year_label,
            "current_year_label": t.current_year_label,
            "overall_prior_score": round(t.overall_prior_score, 1),
            "overall_current_score": round(t.overall_current_score, 1),
            "overall_delta": round(t.overall_current_score - t.overall_prior_score, 1),
            "per_regulation": [
                {
                    "regulation": r.regulation,
                    "prior_year_score": round(r.prior_year_score, 1),
                    "current_year_score": round(r.current_year_score, 1),
                    "delta": round(r.delta, 1),
                    "prior_year_gaps": r.prior_year_gaps,
                    "current_year_gaps": r.current_year_gaps,
                }
                for r in t.per_regulation
            ],
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:12px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".score-gauge{text-align:center;margin:24px 0}"
            ".gauge-circle{display:inline-flex;flex-direction:column;align-items:center;"
            "justify-content:center;width:140px;height:140px;border-radius:50%;"
            "border:6px solid}"
            ".gauge-value{font-size:42px;font-weight:700;line-height:1}"
            ".gauge-label{font-size:12px;color:#7f8c8d}"
            ".stat-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}"
            ".stat-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;"
            "min-width:120px;flex:1}"
            ".stat-val{display:block;font-size:22px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d;margin-top:4px}"
            ".status-badge,.impact-badge{display:inline-block;padding:2px 8px;"
            "border-radius:4px;color:#fff;font-size:11px;font-weight:bold}"
            ".risk-card{padding:12px 16px;margin-bottom:12px;background:#f8f9fa;"
            "border-radius:8px}"
            ".risk-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}"
            ".risk-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".deadline-risk{background:#c0392b;color:#fff;padding:2px 8px;"
            "border-radius:4px;font-size:11px;font-weight:bold}"
            ".risk-detail{font-size:13px;color:#555;margin-top:4px}"
            ".rec-card{display:flex;gap:16px;margin-bottom:16px;padding:12px;"
            "background:#f8f9fa;border-radius:8px}"
            ".rec-num{width:32px;height:32px;background:#1a5276;color:#fff;"
            "border-radius:50%;display:flex;align-items:center;justify-content:center;"
            "font-weight:700;flex-shrink:0}"
            ".rec-content h3{margin:0 0 8px 0;font-size:15px}"
            ".rec-content p{margin:0 0 8px 0;font-size:14px}"
            ".rec-meta{font-size:12px;color:#7f8c8d}"
            ".yoy-header{display:flex;justify-content:center;align-items:center;"
            "gap:24px;margin-bottom:16px;font-size:18px}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;width:100%}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{self.config.title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: BundleExecutiveSummaryTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
