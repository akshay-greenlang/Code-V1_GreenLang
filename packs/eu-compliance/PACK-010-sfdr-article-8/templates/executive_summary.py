"""
ExecutiveSummaryTemplate - Board-level executive summary for SFDR Article 8.

This module implements the executive summary template for PACK-010
SFDR Article 8 products. It generates a concise, board-ready summary
covering fund classification, key metrics, compliance status with
traffic light indicators, risk flags, strategic recommendations,
and upcoming regulatory changes.

Example:
    >>> template = ExecutiveSummaryTemplate()
    >>> data = ExecutiveSummaryData(fund_info=FundInfo(...), ...)
    >>> markdown = template.render_markdown(data.model_dump())
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class FundInfo(BaseModel):
    """Fund identification for executive summary."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN")
    lei: str = Field("", description="LEI")
    management_company: str = Field("", description="Management company")
    domicile: str = Field("", description="Domicile")
    currency: str = Field("EUR", description="Currency")
    nav: Optional[float] = Field(None, ge=0.0, description="NAV")
    aum: Optional[float] = Field(None, ge=0.0, description="AUM")
    inception_date: str = Field("", description="Inception date")


class ClassificationInfo(BaseModel):
    """SFDR classification details."""

    sfdr_article: str = Field("article_8", description="article_8, article_8_plus, article_9")
    classification_date: str = Field("", description="Classification date")
    has_sustainable_investment: bool = Field(False, description="Has sustainable investment objective")
    minimum_sustainable_pct: float = Field(0.0, ge=0.0, le=100.0)
    minimum_taxonomy_pct: float = Field(0.0, ge=0.0, le=100.0)
    pai_consideration: bool = Field(True, description="Considers PAI")
    good_governance: bool = Field(True, description="Applies good governance test")


class KeyMetric(BaseModel):
    """Key performance metric."""

    metric_name: str = Field("", description="Metric name")
    value: str = Field("", description="Current value (formatted)")
    target: str = Field("", description="Target value (formatted)")
    status: str = Field("on_track", description="on_track, at_risk, breached, exceeded")
    trend: str = Field("stable", description="improving, stable, declining")
    commentary: str = Field("", description="Brief commentary")


class ComplianceItem(BaseModel):
    """Compliance status item with traffic light."""

    area: str = Field("", description="Compliance area")
    requirement: str = Field("", description="Requirement description")
    status: str = Field("green", description="green, amber, red")
    last_review: str = Field("", description="Last review date")
    next_deadline: str = Field("", description="Next compliance deadline")
    notes: str = Field("", description="Notes")


class RiskFlag(BaseModel):
    """Risk flag for board attention."""

    risk_id: str = Field("", description="Risk identifier")
    category: str = Field("", description="regulatory, operational, reputational, data")
    title: str = Field("", description="Risk title")
    severity: str = Field("low", description="low, medium, high, critical")
    description: str = Field("", description="Risk description")
    impact: str = Field("", description="Potential impact")
    mitigation: str = Field("", description="Mitigation action")
    owner: str = Field("", description="Risk owner")


class Recommendation(BaseModel):
    """Strategic recommendation."""

    priority: int = Field(0, ge=1, description="Priority rank")
    title: str = Field("", description="Recommendation title")
    description: str = Field("", description="Description")
    expected_impact: str = Field("", description="Expected impact")
    effort: str = Field("low", description="low, medium, high")
    timeline: str = Field("", description="Implementation timeline")
    category: str = Field("", description="Category: compliance, performance, risk, strategic")


class RegulatoryChange(BaseModel):
    """Upcoming regulatory change."""

    regulation: str = Field("", description="Regulation name")
    change_description: str = Field("", description="Description of change")
    effective_date: str = Field("", description="Effective date")
    impact_level: str = Field("low", description="low, medium, high")
    preparation_status: str = Field("not_started", description="not_started, in_progress, ready")
    actions_required: List[str] = Field(default_factory=list)


class ExecutiveSummaryData(BaseModel):
    """Complete input data for executive summary."""

    fund_info: FundInfo
    classification: ClassificationInfo = Field(default_factory=ClassificationInfo)
    key_metrics: List[KeyMetric] = Field(default_factory=list)
    compliance_status: List[ComplianceItem] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    regulatory_outlook: List[RegulatoryChange] = Field(default_factory=list)
    reporting_period: str = Field("", description="Reporting period description")
    prepared_by: str = Field("", description="Prepared by")
    approved_by: str = Field("", description="Approved by")


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class ExecutiveSummaryTemplate:
    """
    Board-level executive summary template for SFDR Article 8 products.

    Generates a concise executive summary with fund classification,
    key metrics, compliance traffic lights, risk flags, recommendations,
    and regulatory outlook.

    Example:
        >>> template = ExecutiveSummaryTemplate()
        >>> md = template.render_markdown(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "executive_summary"
    VERSION = "1.0"

    TRAFFIC_LIGHTS = {
        "green": ("GREEN", "#2ecc71"),
        "amber": ("AMBER", "#f39c12"),
        "red": ("RED", "#e74c3c"),
    }

    SEVERITY_COLORS = {
        "low": "#3498db",
        "medium": "#f39c12",
        "high": "#e67e22",
        "critical": "#e74c3c",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render executive summary in the specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary as Markdown."""
        sections: List[str] = [
            self._md_header(data),
            self._md_classification_card(data),
            self._md_key_metrics(data),
            self._md_compliance_traffic_lights(data),
            self._md_risk_flags(data),
            self._md_recommendations(data),
            self._md_regulatory_outlook(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as HTML."""
        sections: List[str] = [
            self._html_classification_card(data),
            self._html_key_metrics(data),
            self._html_compliance(data),
            self._html_risk_flags(data),
            self._html_recommendations(data),
            self._html_regulatory_outlook(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("SFDR Executive Summary", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary as JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_executive_summary",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_info": data.get("fund_info", {}),
            "classification": data.get("classification", {}),
            "key_metrics": data.get("key_metrics", []),
            "compliance_status": data.get("compliance_status", []),
            "risk_flags": data.get("risk_flags", []),
            "recommendations": data.get("recommendations", []),
            "regulatory_outlook": data.get("regulatory_outlook", []),
            "reporting_period": data.get("reporting_period", ""),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build header."""
        fi = data.get("fund_info", {})
        period = data.get("reporting_period", "")
        return (
            f"# Executive Summary - SFDR Article 8\n\n"
            f"**Fund:** {fi.get('fund_name', 'Unknown')}\n\n"
            f"**Reporting Period:** {period or 'N/A'}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}\n\n"
            f"**Prepared by:** {data.get('prepared_by', 'N/A')}"
        )

    def _md_classification_card(self, data: Dict[str, Any]) -> str:
        """Build fund classification card."""
        fi = data.get("fund_info", {})
        cl = data.get("classification", {})
        article = cl.get("sfdr_article", "article_8")
        nav = fi.get("nav")
        aum = fi.get("aum")

        article_label = {
            "article_8": "Article 8 (Light Green)",
            "article_8_plus": "Article 8+ (Light Green + Sustainable)",
            "article_9": "Article 9 (Dark Green)",
        }.get(article, article)

        lines = [
            "## Fund Classification\n",
            "```",
            f"  Fund:                  {fi.get('fund_name', 'N/A')}",
            f"  SFDR Classification:   {article_label}",
            f"  ISIN:                  {fi.get('isin', '') or 'N/A'}",
            f"  Management Company:    {fi.get('management_company', '') or 'N/A'}",
        ]

        if nav is not None:
            lines.append(f"  NAV:                   {nav:,.2f} {fi.get('currency', 'EUR')}")
        if aum is not None:
            lines.append(f"  AUM:                   {aum:,.2f} {fi.get('currency', 'EUR')}")

        lines.append(f"  Sustainable Inv. Min:  {cl.get('minimum_sustainable_pct', 0.0):.1f}%")
        lines.append(f"  Taxonomy Min:          {cl.get('minimum_taxonomy_pct', 0.0):.1f}%")
        lines.append(f"  PAI Consideration:     {'Yes' if cl.get('pai_consideration') else 'No'}")
        lines.append(f"  Good Governance:       {'Yes' if cl.get('good_governance') else 'No'}")
        lines.append("```")

        return "\n".join(lines)

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Build key metrics table."""
        metrics = data.get("key_metrics", [])

        lines = [
            "## Key Metrics\n",
            "| Metric | Value | Target | Status | Trend |",
            "|--------|-------|--------|--------|-------|",
        ]

        status_icons = {
            "on_track": "[OK]",
            "at_risk": "[RISK]",
            "breached": "[BREACH]",
            "exceeded": "[EXCEED]",
        }
        trend_icons = {"improving": "^", "stable": "=", "declining": "v"}

        for m in metrics:
            status = status_icons.get(m.get("status", ""), m.get("status", ""))
            trend = trend_icons.get(m.get("trend", ""), m.get("trend", ""))
            lines.append(
                f"| {m.get('metric_name', '')} | {m.get('value', '')} | "
                f"{m.get('target', '')} | {status} | {trend} |"
            )

        if not metrics:
            lines.append("| *No metrics defined* | | | | |")

        return "\n".join(lines)

    def _md_compliance_traffic_lights(self, data: Dict[str, Any]) -> str:
        """Build compliance traffic light section."""
        items = data.get("compliance_status", [])

        lines = [
            "## Compliance Status\n",
            "| Area | Status | Last Review | Next Deadline | Notes |",
            "|------|--------|-------------|---------------|-------|",
        ]

        for item in items:
            status = item.get("status", "green")
            label = self.TRAFFIC_LIGHTS.get(status, ("UNKNOWN", "#95a5a6"))[0]
            lines.append(
                f"| {item.get('area', '')} | [{label}] | "
                f"{item.get('last_review', '')} | "
                f"{item.get('next_deadline', '')} | "
                f"{item.get('notes', '') or '-'} |"
            )

        if not items:
            lines.append("| *No compliance items* | | | | |")

        # Summary counts
        green_count = sum(1 for i in items if i.get("status") == "green")
        amber_count = sum(1 for i in items if i.get("status") == "amber")
        red_count = sum(1 for i in items if i.get("status") == "red")
        total = len(items)

        lines.append(f"\n**Summary:** {green_count} GREEN / {amber_count} AMBER / {red_count} RED "
                      f"(of {total} areas)")

        return "\n".join(lines)

    def _md_risk_flags(self, data: Dict[str, Any]) -> str:
        """Build risk flags section."""
        risks = data.get("risk_flags", [])
        if not risks:
            return "## Risk Flags\n\nNo active risk flags."

        lines = ["## Risk Flags\n"]

        # Summary by severity
        critical = [r for r in risks if r.get("severity") == "critical"]
        high = [r for r in risks if r.get("severity") == "high"]

        if critical:
            lines.append(f"**CRITICAL RISKS: {len(critical)}**\n")
        if high:
            lines.append(f"**HIGH RISKS: {len(high)}**\n")

        lines.append("| ID | Category | Title | Severity | Impact | Owner |")
        lines.append("|----|----------|-------|----------|--------|-------|")

        for r in risks:
            sev = r.get("severity", "low").upper()
            lines.append(
                f"| {r.get('risk_id', '')} | {r.get('category', '')} | "
                f"{r.get('title', '')} | [{sev}] | "
                f"{r.get('impact', '')} | {r.get('owner', '')} |"
            )

        # Detail for critical/high
        for r in risks:
            if r.get("severity") in ("critical", "high"):
                lines.append(f"\n### {r.get('risk_id', '')}: {r.get('title', '')}\n")
                lines.append(f"{r.get('description', '')}\n")
                mitigation = r.get("mitigation", "")
                if mitigation:
                    lines.append(f"**Mitigation:** {mitigation}")

        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Build strategic recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""

        lines = [
            "## Strategic Recommendations\n",
            "| Priority | Title | Category | Effort | Timeline |",
            "|----------|-------|----------|--------|----------|",
        ]

        for r in sorted(recs, key=lambda x: x.get("priority", 99)):
            lines.append(
                f"| {r.get('priority', 0)} | {r.get('title', '')} | "
                f"{r.get('category', '')} | {r.get('effort', '').upper()} | "
                f"{r.get('timeline', '')} |"
            )

        # Detailed descriptions
        lines.append("")
        for r in sorted(recs, key=lambda x: x.get("priority", 99)):
            lines.append(f"### {r.get('priority', 0)}. {r.get('title', '')}\n")
            lines.append(f"{r.get('description', '')}\n")
            impact = r.get("expected_impact", "")
            if impact:
                lines.append(f"**Expected Impact:** {impact}\n")

        return "\n".join(lines)

    def _md_regulatory_outlook(self, data: Dict[str, Any]) -> str:
        """Build regulatory outlook section."""
        changes = data.get("regulatory_outlook", [])
        if not changes:
            return ""

        lines = [
            "## Regulatory Outlook\n",
            "| Regulation | Change | Effective Date | Impact | Status |",
            "|------------|--------|----------------|--------|--------|",
        ]

        for c in changes:
            status_label = {
                "not_started": "NOT STARTED",
                "in_progress": "IN PROGRESS",
                "ready": "READY",
            }.get(c.get("preparation_status", ""), "UNKNOWN")
            lines.append(
                f"| {c.get('regulation', '')} | {c.get('change_description', '')} | "
                f"{c.get('effective_date', '')} | {c.get('impact_level', '').upper()} | "
                f"{status_label} |"
            )

        # Detailed actions
        for c in changes:
            actions = c.get("actions_required", [])
            if actions:
                lines.append(f"\n### {c.get('regulation', '')}\n")
                lines.append("Actions required:")
                for a in actions:
                    lines.append(f"- {a}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_classification_card(self, data: Dict[str, Any]) -> str:
        """Build HTML classification card."""
        fi = data.get("fund_info", {})
        cl = data.get("classification", {})
        article = cl.get("sfdr_article", "article_8")

        article_label = {
            "article_8": "Article 8",
            "article_8_plus": "Article 8+",
            "article_9": "Article 9",
        }.get(article, article)

        article_color = {
            "article_8": "#27ae60",
            "article_8_plus": "#2ecc71",
            "article_9": "#16a085",
        }.get(article, "#2c3e50")

        nav = fi.get("nav")
        nav_str = f"{nav:,.2f} {fi.get('currency', 'EUR')}" if nav else "N/A"

        return (
            '<div class="card">'
            "<h2>Fund Classification</h2>"
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
            '<div>'
            f'<div style="text-align:center;padding:20px;background:{article_color};'
            f'color:white;border-radius:8px;font-size:1.4em;font-weight:bold;">'
            f"SFDR {_esc(article_label)}</div>"
            "</div><div>"
            f"<p><strong>Fund:</strong> {_esc(fi.get('fund_name', ''))}</p>"
            f"<p><strong>ISIN:</strong> {_esc(fi.get('isin', '') or 'N/A')}</p>"
            f"<p><strong>NAV:</strong> {_esc(nav_str)}</p>"
            f"<p><strong>Min. Sustainable:</strong> {cl.get('minimum_sustainable_pct', 0.0):.1f}%</p>"
            f"<p><strong>Min. Taxonomy:</strong> {cl.get('minimum_taxonomy_pct', 0.0):.1f}%</p>"
            "</div></div></div>"
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Build HTML key metrics."""
        metrics = data.get("key_metrics", [])
        parts = ['<div class="card"><h2>Key Metrics</h2>']

        if metrics:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Metric</th><th>Value</th><th>Target</th>"
                "<th>Status</th><th>Trend</th></tr>"
            )
            status_colors = {
                "on_track": "#2ecc71",
                "at_risk": "#f39c12",
                "breached": "#e74c3c",
                "exceeded": "#3498db",
            }
            for m in metrics:
                status = m.get("status", "on_track")
                color = status_colors.get(status, "#2c3e50")
                label = status.upper().replace("_", " ")
                trend = m.get("trend", "stable")
                parts.append(
                    f"<tr><td>{_esc(m.get('metric_name', ''))}</td>"
                    f"<td><strong>{_esc(m.get('value', ''))}</strong></td>"
                    f"<td>{_esc(m.get('target', ''))}</td>"
                    f'<td style="color:{color};font-weight:bold;">{label}</td>'
                    f"<td>{_esc(trend.capitalize())}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_compliance(self, data: Dict[str, Any]) -> str:
        """Build HTML compliance traffic lights."""
        items = data.get("compliance_status", [])
        parts = ['<div class="card"><h2>Compliance Status</h2>']

        if items:
            parts.append('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;">')
            for item in items:
                status = item.get("status", "green")
                color = self.TRAFFIC_LIGHTS.get(status, ("", "#95a5a6"))[1]
                parts.append(
                    f'<div style="padding:12px;border-left:5px solid {color};'
                    f'background:#f8f9fa;border-radius:0 6px 6px 0;">'
                    f'<div style="font-weight:bold;">{_esc(item.get("area", ""))}</div>'
                    f'<div style="color:{color};font-weight:bold;font-size:0.9em;">'
                    f'{self.TRAFFIC_LIGHTS.get(status, ("UNKNOWN",))[0]}</div>'
                    f'<div style="font-size:0.85em;color:#7f8c8d;">'
                    f'Next: {_esc(item.get("next_deadline", ""))}</div>'
                    f"</div>"
                )
            parts.append("</div>")

        parts.append("</div>")
        return "".join(parts)

    def _html_risk_flags(self, data: Dict[str, Any]) -> str:
        """Build HTML risk flags."""
        risks = data.get("risk_flags", [])
        parts = ['<div class="card"><h2>Risk Flags</h2>']

        if risks:
            for r in risks:
                severity = r.get("severity", "low")
                color = self.SEVERITY_COLORS.get(severity, "#2c3e50")
                parts.append(
                    f'<div style="border-left:4px solid {color};padding:10px;'
                    f'margin:8px 0;background:#fdfdfd;border-radius:0 4px 4px 0;">'
                    f'<strong style="color:{color};">[{severity.upper()}] '
                    f'{_esc(r.get("title", ""))}</strong>'
                    f"<p>{_esc(r.get('description', ''))}</p>"
                    f"<p><strong>Owner:</strong> {_esc(r.get('owner', ''))}</p>"
                    f"</div>"
                )
        else:
            parts.append("<p>No active risk flags.</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Build HTML recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""

        parts = ['<div class="card"><h2>Strategic Recommendations</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>#</th><th>Title</th><th>Category</th>"
            "<th>Effort</th><th>Timeline</th></tr>"
        )
        for r in sorted(recs, key=lambda x: x.get("priority", 99)):
            parts.append(
                f"<tr><td>{r.get('priority', 0)}</td>"
                f"<td><strong>{_esc(r.get('title', ''))}</strong><br>"
                f"<small>{_esc(r.get('description', ''))}</small></td>"
                f"<td>{_esc(r.get('category', ''))}</td>"
                f"<td>{_esc(r.get('effort', '').upper())}</td>"
                f"<td>{_esc(r.get('timeline', ''))}</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_regulatory_outlook(self, data: Dict[str, Any]) -> str:
        """Build HTML regulatory outlook."""
        changes = data.get("regulatory_outlook", [])
        if not changes:
            return ""

        parts = ['<div class="card"><h2>Regulatory Outlook</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>Regulation</th><th>Change</th><th>Effective</th>"
            "<th>Impact</th><th>Status</th></tr>"
        )
        for c in changes:
            impact = c.get("impact_level", "low")
            impact_color = {"low": "#3498db", "medium": "#f39c12", "high": "#e74c3c"}.get(
                impact, "#2c3e50"
            )
            status = c.get("preparation_status", "not_started")
            status_color = {
                "not_started": "#e74c3c",
                "in_progress": "#f39c12",
                "ready": "#2ecc71",
            }.get(status, "#2c3e50")
            status_label = status.upper().replace("_", " ")
            parts.append(
                f"<tr><td>{_esc(c.get('regulation', ''))}</td>"
                f"<td>{_esc(c.get('change_description', ''))}</td>"
                f"<td>{_esc(c.get('effective_date', ''))}</td>"
                f'<td style="color:{impact_color};">{impact.upper()}</td>'
                f'<td style="color:{status_color};font-weight:bold;">{status_label}</td></tr>'
            )
        parts.append("</table></div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown footer."""
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap in HTML document."""
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px auto; "
            "color: #2c3e50; line-height: 1.5; max-width: 1100px; background: #f4f6f7; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 0; }\n"
            ".card { background: white; border-radius: 8px; padding: 20px; "
            "margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 30px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f"<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
