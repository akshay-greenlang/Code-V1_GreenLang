"""
PortfolioESGDashboardTemplate - Interactive-style ESG metrics dashboard.

This module implements the portfolio ESG dashboard template for PACK-010
SFDR Article 8 products. It generates a comprehensive dashboard view of
ESG metrics including fund overview, ESG scores, taxonomy alignment,
carbon metrics, sector allocation, PAI summary, commitment tracking,
and compliance alerts.

Uses ASCII art/text gauges for Markdown output and styled divs for HTML.

Example:
    >>> template = PortfolioESGDashboardTemplate()
    >>> data = DashboardData(fund_overview=FundOverview(...), ...)
    >>> html = template.render_html(data.model_dump())
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

class FundOverview(BaseModel):
    """Fund overview information."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    nav: Optional[float] = Field(None, ge=0.0, description="Net Asset Value")
    currency: str = Field("EUR", description="Currency")
    inception_date: str = Field("", description="Fund inception date")
    sfdr_classification: str = Field("article_8", description="SFDR classification")
    benchmark: str = Field("", description="Benchmark name")
    total_holdings: int = Field(0, ge=0, description="Total number of holdings")
    as_of_date: str = Field("", description="Data as-of date")


class ESGScores(BaseModel):
    """ESG score metrics."""

    overall_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall ESG score")
    environmental_score: float = Field(0.0, ge=0.0, le=100.0, description="E score")
    social_score: float = Field(0.0, ge=0.0, le=100.0, description="S score")
    governance_score: float = Field(0.0, ge=0.0, le=100.0, description="G score")
    benchmark_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Benchmark ESG score")
    coverage_pct: float = Field(100.0, ge=0.0, le=100.0, description="ESG coverage %")
    rating_provider: str = Field("", description="ESG rating provider")
    trend: str = Field("stable", description="improving, stable, declining")


class TaxonomyAlignment(BaseModel):
    """Taxonomy alignment metrics."""

    eligible_pct: float = Field(0.0, ge=0.0, le=100.0, description="Taxonomy-eligible %")
    aligned_pct: float = Field(0.0, ge=0.0, le=100.0, description="Taxonomy-aligned %")
    by_objective: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="By objective: {name, eligible_pct, aligned_pct}",
    )
    commitment_pct: float = Field(0.0, ge=0.0, le=100.0, description="Commitment target %")
    transitional_pct: float = Field(0.0, ge=0.0, le=100.0, description="Transitional %")
    enabling_pct: float = Field(0.0, ge=0.0, le=100.0, description="Enabling %")


class CarbonMetrics(BaseModel):
    """Carbon/climate metrics."""

    carbon_footprint: float = Field(0.0, ge=0.0, description="tCO2e per EUR M invested")
    weighted_avg_carbon_intensity: float = Field(
        0.0, ge=0.0, description="tCO2e/EUR M revenue (WACI)"
    )
    total_financed_emissions: float = Field(0.0, ge=0.0, description="Total tCO2e")
    scope1_pct: float = Field(0.0, ge=0.0, le=100.0, description="Scope 1 share %")
    scope2_pct: float = Field(0.0, ge=0.0, le=100.0, description="Scope 2 share %")
    scope3_pct: float = Field(0.0, ge=0.0, le=100.0, description="Scope 3 share %")
    benchmark_waci: Optional[float] = Field(None, ge=0.0, description="Benchmark WACI")
    fossil_fuel_exposure_pct: float = Field(0.0, ge=0.0, le=100.0, description="Fossil fuel %")
    coverage_pct: float = Field(100.0, ge=0.0, le=100.0, description="Carbon data coverage %")


class SectorEntry(BaseModel):
    """Sector allocation entry."""

    sector: str = Field("", description="Sector name")
    weight_pct: float = Field(0.0, ge=0.0, le=100.0, description="Weight %")
    esg_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Sector ESG score")
    carbon_intensity: Optional[float] = Field(None, ge=0.0, description="Sector carbon intensity")
    taxonomy_aligned_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Taxonomy aligned in sector %"
    )


class PAIDashboardSummary(BaseModel):
    """PAI summary for dashboard."""

    total_indicators_tracked: int = Field(18, ge=0, description="Total PAI indicators tracked")
    indicators_improving: int = Field(0, ge=0, description="Count improving YoY")
    indicators_stable: int = Field(0, ge=0, description="Count stable")
    indicators_worsening: int = Field(0, ge=0, description="Count worsening")
    key_highlights: List[str] = Field(default_factory=list, description="Key PAI highlights")


class CommitmentItem(BaseModel):
    """Commitment tracking item."""

    commitment: str = Field("", description="Commitment description")
    target: str = Field("", description="Target value/description")
    current: str = Field("", description="Current status")
    on_track: bool = Field(True, description="Whether on track")
    deadline: str = Field("", description="Target deadline")


class ComplianceAlert(BaseModel):
    """Compliance alert."""

    alert_type: str = Field("info", description="info, warning, error, critical")
    title: str = Field("", description="Alert title")
    description: str = Field("", description="Alert description")
    action_required: str = Field("", description="Required action")
    deadline: str = Field("", description="Action deadline")


class DashboardData(BaseModel):
    """Complete input data for ESG dashboard."""

    fund_overview: FundOverview
    esg_scores: ESGScores = Field(default_factory=ESGScores)
    taxonomy_alignment: TaxonomyAlignment = Field(default_factory=TaxonomyAlignment)
    carbon_metrics: CarbonMetrics = Field(default_factory=CarbonMetrics)
    sector_allocation: List[SectorEntry] = Field(default_factory=list)
    pai_summary: PAIDashboardSummary = Field(default_factory=PAIDashboardSummary)
    commitment_tracking: List[CommitmentItem] = Field(default_factory=list)
    alerts: List[ComplianceAlert] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class PortfolioESGDashboardTemplate:
    """
    Portfolio ESG dashboard template for SFDR Article 8 products.

    Generates an interactive-style dashboard with fund overview, ESG scores,
    taxonomy alignment, carbon metrics, sector allocation, PAI summary,
    commitment tracking, and compliance alerts.

    Example:
        >>> template = PortfolioESGDashboardTemplate()
        >>> html = template.render_html(data)
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "portfolio_esg_dashboard"
    VERSION = "1.0"

    ALERT_COLORS = {
        "info": ("#3498db", "INFO"),
        "warning": ("#f39c12", "WARNING"),
        "error": ("#e74c3c", "ERROR"),
        "critical": ("#c0392b", "CRITICAL"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PortfolioESGDashboardTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render dashboard in the specified format."""
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
        """Render ESG dashboard as Markdown with ASCII gauges."""
        sections: List[str] = [
            self._md_header(data),
            self._md_fund_overview(data),
            self._md_esg_scores(data),
            self._md_taxonomy_alignment(data),
            self._md_carbon_metrics(data),
            self._md_sector_allocation(data),
            self._md_pai_summary(data),
            self._md_commitment_tracker(data),
            self._md_alerts(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESG dashboard as styled HTML."""
        sections: List[str] = [
            self._html_fund_overview(data),
            self._html_esg_scores(data),
            self._html_taxonomy_alignment(data),
            self._html_carbon_metrics(data),
            self._html_sector_allocation(data),
            self._html_pai_summary(data),
            self._html_commitment_tracker(data),
            self._html_alerts(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html("Portfolio ESG Dashboard", body, provenance_hash)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESG dashboard as structured JSON."""
        report: Dict[str, Any] = {
            "report_type": "sfdr_portfolio_esg_dashboard",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_overview": data.get("fund_overview", {}),
            "esg_scores": data.get("esg_scores", {}),
            "taxonomy_alignment": data.get("taxonomy_alignment", {}),
            "carbon_metrics": data.get("carbon_metrics", {}),
            "sector_allocation": data.get("sector_allocation", []),
            "pai_summary": data.get("pai_summary", {}),
            "commitment_tracking": data.get("commitment_tracking", []),
            "alerts": data.get("alerts", []),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown header."""
        fo = data.get("fund_overview", {})
        return (
            f"# Portfolio ESG Dashboard\n\n"
            f"**Fund:** {fo.get('fund_name', 'Unknown')}\n\n"
            f"**As of:** {fo.get('as_of_date', 'N/A')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_fund_overview(self, data: Dict[str, Any]) -> str:
        """Build fund overview card."""
        fo = data.get("fund_overview", {})
        nav = fo.get("nav")
        nav_str = f"{nav:,.2f} {fo.get('currency', 'EUR')}" if nav else "N/A"
        classification = self._format_classification(fo.get("sfdr_classification", ""))

        lines = [
            "## Fund Overview\n",
            "```",
            f"  Fund:           {fo.get('fund_name', 'N/A')}",
            f"  ISIN:           {fo.get('isin', '') or 'N/A'}",
            f"  Classification: {classification}",
            f"  NAV:            {nav_str}",
            f"  Holdings:       {fo.get('total_holdings', 0)}",
            f"  Benchmark:      {fo.get('benchmark', '') or 'None'}",
            f"  Inception:      {fo.get('inception_date', '') or 'N/A'}",
            "```",
        ]
        return "\n".join(lines)

    def _md_esg_scores(self, data: Dict[str, Any]) -> str:
        """Build ESG score section with ASCII gauges."""
        esg = data.get("esg_scores", {})
        overall = esg.get("overall_score", 0.0)
        e_score = esg.get("environmental_score", 0.0)
        s_score = esg.get("social_score", 0.0)
        g_score = esg.get("governance_score", 0.0)
        bench = esg.get("benchmark_score")
        trend = esg.get("trend", "stable")

        trend_arrow = {"improving": "^", "stable": "=", "declining": "v"}.get(trend, "=")

        lines = [
            "## ESG Scores\n",
            "```",
            f"  Overall:       {self._ascii_gauge(overall)} {overall:.1f}/100  [{trend_arrow} {trend}]",
            f"  Environmental: {self._ascii_gauge(e_score)} {e_score:.1f}/100",
            f"  Social:        {self._ascii_gauge(s_score)} {s_score:.1f}/100",
            f"  Governance:    {self._ascii_gauge(g_score)} {g_score:.1f}/100",
        ]
        if bench is not None:
            diff = overall - bench
            lines.append(f"  Benchmark:     {bench:.1f}/100  (diff: {diff:+.1f})")
        lines.append(f"  Coverage:      {esg.get('coverage_pct', 100.0):.0f}%")
        lines.append(f"  Provider:      {esg.get('rating_provider', 'N/A')}")
        lines.append("```")

        return "\n".join(lines)

    def _md_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Build taxonomy alignment section."""
        ta = data.get("taxonomy_alignment", {})
        eligible = ta.get("eligible_pct", 0.0)
        aligned = ta.get("aligned_pct", 0.0)
        commitment = ta.get("commitment_pct", 0.0)
        objectives = ta.get("by_objective", [])

        lines = [
            "## Taxonomy Alignment\n",
            "```",
            f"  Eligible:      {self._ascii_gauge(eligible)} {eligible:.1f}%",
            f"  Aligned:       {self._ascii_gauge(aligned)} {aligned:.1f}%",
            f"  Commitment:    {commitment:.1f}%",
            f"  Transitional:  {ta.get('transitional_pct', 0.0):.1f}%",
            f"  Enabling:      {ta.get('enabling_pct', 0.0):.1f}%",
            "```\n",
        ]

        if objectives:
            lines.append("### By Environmental Objective\n")
            lines.append("| Objective | Eligible | Aligned |")
            lines.append("|-----------|----------|---------|")
            for obj in objectives:
                lines.append(
                    f"| {obj.get('name', '')} | "
                    f"{obj.get('eligible_pct', 0.0):.1f}% | "
                    f"{obj.get('aligned_pct', 0.0):.1f}% |"
                )

        return "\n".join(lines)

    def _md_carbon_metrics(self, data: Dict[str, Any]) -> str:
        """Build carbon metrics section."""
        cm = data.get("carbon_metrics", {})
        bench_waci = cm.get("benchmark_waci")

        lines = [
            "## Carbon Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Carbon Footprint | {cm.get('carbon_footprint', 0.0):.2f} tCO2e/EUR M |",
            f"| WACI | {cm.get('weighted_avg_carbon_intensity', 0.0):.2f} tCO2e/EUR M revenue |",
            f"| Total Financed Emissions | {cm.get('total_financed_emissions', 0.0):,.0f} tCO2e |",
            f"| Fossil Fuel Exposure | {cm.get('fossil_fuel_exposure_pct', 0.0):.1f}% |",
            f"| Data Coverage | {cm.get('coverage_pct', 100.0):.0f}% |",
        ]

        if bench_waci is not None:
            diff = cm.get("weighted_avg_carbon_intensity", 0.0) - bench_waci
            lines.append(f"| Benchmark WACI | {bench_waci:.2f} (diff: {diff:+.2f}) |")

        # Scope breakdown
        lines.append(f"\n### Emission Scope Breakdown\n")
        lines.append("```")
        lines.append(f"  Scope 1: {self._ascii_bar(cm.get('scope1_pct', 0.0), 40)} {cm.get('scope1_pct', 0.0):.1f}%")
        lines.append(f"  Scope 2: {self._ascii_bar(cm.get('scope2_pct', 0.0), 40)} {cm.get('scope2_pct', 0.0):.1f}%")
        lines.append(f"  Scope 3: {self._ascii_bar(cm.get('scope3_pct', 0.0), 40)} {cm.get('scope3_pct', 0.0):.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_sector_allocation(self, data: Dict[str, Any]) -> str:
        """Build sector allocation table."""
        sectors = data.get("sector_allocation", [])
        lines = [
            "## Sector Allocation\n",
            "| Sector | Weight | ESG Score | Carbon Intensity | Tax. Aligned |",
            "|--------|--------|-----------|------------------|--------------|",
        ]

        for s in sectors:
            esg = f"{s.get('esg_score', 0):.1f}" if s.get("esg_score") is not None else "N/A"
            ci = f"{s.get('carbon_intensity', 0):.1f}" if s.get("carbon_intensity") is not None else "N/A"
            ta = f"{s.get('taxonomy_aligned_pct', 0):.1f}%" if s.get("taxonomy_aligned_pct") is not None else "N/A"
            lines.append(
                f"| {s.get('sector', '')} | {s.get('weight_pct', 0.0):.1f}% | "
                f"{esg} | {ci} | {ta} |"
            )

        if not sectors:
            lines.append("| *No sector data* | | | | |")

        return "\n".join(lines)

    def _md_pai_summary(self, data: Dict[str, Any]) -> str:
        """Build PAI summary tiles."""
        pai = data.get("pai_summary", {})
        total = pai.get("total_indicators_tracked", 18)
        improving = pai.get("indicators_improving", 0)
        stable = pai.get("indicators_stable", 0)
        worsening = pai.get("indicators_worsening", 0)
        highlights = pai.get("key_highlights", [])

        lines = [
            "## PAI Indicator Summary\n",
            "```",
            f"  Total Tracked:  {total}",
            f"  Improving:      {improving}  [{'#' * improving}{'-' * (total - improving)}]",
            f"  Stable:         {stable}",
            f"  Worsening:      {worsening}",
            "```\n",
        ]

        if highlights:
            lines.append("### Key Highlights\n")
            for h in highlights:
                lines.append(f"- {h}")

        return "\n".join(lines)

    def _md_commitment_tracker(self, data: Dict[str, Any]) -> str:
        """Build commitment tracking table."""
        commitments = data.get("commitment_tracking", [])
        if not commitments:
            return ""

        lines = [
            "## Commitment Tracker\n",
            "| Commitment | Target | Current | Status | Deadline |",
            "|------------|--------|---------|--------|----------|",
        ]

        for c in commitments:
            status = "[ON TRACK]" if c.get("on_track") else "[AT RISK]"
            lines.append(
                f"| {c.get('commitment', '')} | {c.get('target', '')} | "
                f"{c.get('current', '')} | {status} | {c.get('deadline', '')} |"
            )

        return "\n".join(lines)

    def _md_alerts(self, data: Dict[str, Any]) -> str:
        """Build compliance alerts section."""
        alerts = data.get("alerts", [])
        if not alerts:
            return ""

        lines = ["## Compliance Alerts\n"]

        for a in alerts:
            alert_type = a.get("alert_type", "info").upper()
            lines.append(
                f"### [{alert_type}] {a.get('title', 'Alert')}\n"
            )
            lines.append(f"{a.get('description', '')}\n")
            action = a.get("action_required", "")
            if action:
                lines.append(f"**Action Required:** {action}\n")
            deadline = a.get("deadline", "")
            if deadline:
                lines.append(f"**Deadline:** {deadline}\n")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_fund_overview(self, data: Dict[str, Any]) -> str:
        """Build HTML fund overview card."""
        fo = data.get("fund_overview", {})
        nav = fo.get("nav")
        nav_str = f"{nav:,.2f} {fo.get('currency', 'EUR')}" if nav else "N/A"

        return (
            '<div class="card">'
            "<h2>Fund Overview</h2>"
            '<div class="card-grid">'
            f'<div class="metric-box"><span class="label">Fund</span>'
            f'<span class="value">{_esc(fo.get("fund_name", ""))}</span></div>'
            f'<div class="metric-box"><span class="label">Classification</span>'
            f'<span class="value">{_esc(self._format_classification(fo.get("sfdr_classification", "")))}</span></div>'
            f'<div class="metric-box"><span class="label">NAV</span>'
            f'<span class="value">{_esc(nav_str)}</span></div>'
            f'<div class="metric-box"><span class="label">Holdings</span>'
            f'<span class="value">{fo.get("total_holdings", 0)}</span></div>'
            "</div></div>"
        )

    def _html_esg_scores(self, data: Dict[str, Any]) -> str:
        """Build HTML ESG score gauges."""
        esg = data.get("esg_scores", {})
        scores = [
            ("Overall", esg.get("overall_score", 0.0)),
            ("Environmental", esg.get("environmental_score", 0.0)),
            ("Social", esg.get("social_score", 0.0)),
            ("Governance", esg.get("governance_score", 0.0)),
        ]

        parts = ['<div class="card"><h2>ESG Scores</h2><div class="card-grid">']
        for label, score in scores:
            color = self._score_color(score)
            parts.append(
                f'<div class="metric-box">'
                f'<span class="label">{_esc(label)}</span>'
                f'<div class="gauge-container">'
                f'<div class="gauge-fill" style="width:{score}%;background:{color};"></div>'
                f"</div>"
                f'<span class="value" style="color:{color};">{score:.1f}/100</span>'
                f"</div>"
            )
        parts.append("</div></div>")
        return "".join(parts)

    def _html_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Build HTML taxonomy alignment donut-style display."""
        ta = data.get("taxonomy_alignment", {})
        eligible = ta.get("eligible_pct", 0.0)
        aligned = ta.get("aligned_pct", 0.0)

        parts = ['<div class="card"><h2>Taxonomy Alignment</h2>']
        parts.append('<div class="card-grid">')
        parts.append(
            f'<div class="metric-box">'
            f'<span class="label">Eligible</span>'
            f'<div class="gauge-container">'
            f'<div class="gauge-fill" style="width:{eligible}%;background:#3498db;"></div>'
            f"</div>"
            f'<span class="value">{eligible:.1f}%</span></div>'
        )
        parts.append(
            f'<div class="metric-box">'
            f'<span class="label">Aligned</span>'
            f'<div class="gauge-container">'
            f'<div class="gauge-fill" style="width:{aligned}%;background:#2ecc71;"></div>'
            f"</div>"
            f'<span class="value">{aligned:.1f}%</span></div>'
        )
        parts.append("</div>")

        objectives = ta.get("by_objective", [])
        if objectives:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Objective</th><th>Eligible</th><th>Aligned</th></tr>")
            for obj in objectives:
                parts.append(
                    f"<tr><td>{_esc(obj.get('name', ''))}</td>"
                    f"<td>{obj.get('eligible_pct', 0.0):.1f}%</td>"
                    f"<td>{obj.get('aligned_pct', 0.0):.1f}%</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_carbon_metrics(self, data: Dict[str, Any]) -> str:
        """Build HTML carbon metrics section."""
        cm = data.get("carbon_metrics", {})
        parts = ['<div class="card"><h2>Carbon Metrics</h2><div class="card-grid">']

        metrics = [
            ("Carbon Footprint", f"{cm.get('carbon_footprint', 0.0):.2f}", "tCO2e/EUR M"),
            ("WACI", f"{cm.get('weighted_avg_carbon_intensity', 0.0):.2f}", "tCO2e/EUR M rev"),
            ("Total Financed", f"{cm.get('total_financed_emissions', 0.0):,.0f}", "tCO2e"),
            ("Fossil Fuel", f"{cm.get('fossil_fuel_exposure_pct', 0.0):.1f}", "%"),
        ]
        for label, value, unit in metrics:
            parts.append(
                f'<div class="metric-box">'
                f'<span class="label">{_esc(label)}</span>'
                f'<span class="value">{value} <small>{_esc(unit)}</small></span></div>'
            )

        parts.append("</div>")

        # Scope bars
        scope_items = [
            ("Scope 1", cm.get("scope1_pct", 0.0), "#e74c3c"),
            ("Scope 2", cm.get("scope2_pct", 0.0), "#f39c12"),
            ("Scope 3", cm.get("scope3_pct", 0.0), "#3498db"),
        ]
        for label, pct, color in scope_items:
            bar_w = max(int(pct * 3), 0)
            parts.append(
                f'<div style="margin:6px 0;">'
                f'<span style="display:inline-block;width:80px;">{label}</span>'
                f'<div style="display:inline-block;background:{color};'
                f'width:{bar_w}px;height:16px;border-radius:3px;'
                f'vertical-align:middle;margin-right:8px;"></div>'
                f"<span>{pct:.1f}%</span></div>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _html_sector_allocation(self, data: Dict[str, Any]) -> str:
        """Build HTML sector allocation table."""
        sectors = data.get("sector_allocation", [])
        parts = ['<div class="card"><h2>Sector Allocation</h2>']
        if sectors:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Sector</th><th>Weight</th><th>ESG</th>"
                "<th>Carbon Int.</th><th>Tax. Aligned</th></tr>"
            )
            for s in sectors:
                esg = f"{s.get('esg_score', 0):.1f}" if s.get("esg_score") is not None else "N/A"
                ci = f"{s.get('carbon_intensity', 0):.1f}" if s.get("carbon_intensity") is not None else "N/A"
                ta = f"{s.get('taxonomy_aligned_pct', 0):.1f}%" if s.get("taxonomy_aligned_pct") is not None else "N/A"
                parts.append(
                    f"<tr><td>{_esc(s.get('sector', ''))}</td>"
                    f"<td>{s.get('weight_pct', 0):.1f}%</td>"
                    f"<td>{esg}</td><td>{ci}</td><td>{ta}</td></tr>"
                )
            parts.append("</table>")
        parts.append("</div>")
        return "".join(parts)

    def _html_pai_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML PAI summary tiles."""
        pai = data.get("pai_summary", {})
        parts = ['<div class="card"><h2>PAI Summary</h2><div class="card-grid">']
        tiles = [
            ("Total Tracked", str(pai.get("total_indicators_tracked", 18)), "#2c3e50"),
            ("Improving", str(pai.get("indicators_improving", 0)), "#2ecc71"),
            ("Stable", str(pai.get("indicators_stable", 0)), "#3498db"),
            ("Worsening", str(pai.get("indicators_worsening", 0)), "#e74c3c"),
        ]
        for label, value, color in tiles:
            parts.append(
                f'<div class="metric-box">'
                f'<span class="label">{_esc(label)}</span>'
                f'<span class="value" style="color:{color};font-size:1.8em;">{value}</span>'
                f"</div>"
            )
        parts.append("</div>")

        highlights = pai.get("key_highlights", [])
        if highlights:
            parts.append("<h3>Key Highlights</h3><ul>")
            for h in highlights:
                parts.append(f"<li>{_esc(h)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_commitment_tracker(self, data: Dict[str, Any]) -> str:
        """Build HTML commitment tracker."""
        commitments = data.get("commitment_tracking", [])
        if not commitments:
            return ""

        parts = ['<div class="card"><h2>Commitment Tracker</h2>']
        parts.append('<table class="data-table">')
        parts.append(
            "<tr><th>Commitment</th><th>Target</th><th>Current</th>"
            "<th>Status</th><th>Deadline</th></tr>"
        )
        for c in commitments:
            on_track = c.get("on_track", True)
            color = "#2ecc71" if on_track else "#e74c3c"
            status = "ON TRACK" if on_track else "AT RISK"
            parts.append(
                f"<tr><td>{_esc(c.get('commitment', ''))}</td>"
                f"<td>{_esc(c.get('target', ''))}</td>"
                f"<td>{_esc(c.get('current', ''))}</td>"
                f'<td style="color:{color};font-weight:bold;">{status}</td>'
                f"<td>{_esc(c.get('deadline', ''))}</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_alerts(self, data: Dict[str, Any]) -> str:
        """Build HTML compliance alerts."""
        alerts = data.get("alerts", [])
        if not alerts:
            return ""

        parts = ['<div class="card"><h2>Compliance Alerts</h2>']
        for a in alerts:
            alert_type = a.get("alert_type", "info")
            color, label = self.ALERT_COLORS.get(alert_type, ("#3498db", "INFO"))
            parts.append(
                f'<div style="border-left:4px solid {color};padding:10px;'
                f'margin:10px 0;background:#fdfdfd;border-radius:0 4px 4px 0;">'
                f'<strong style="color:{color};">[{label}] {_esc(a.get("title", ""))}</strong>'
                f"<p>{_esc(a.get('description', ''))}</p>"
            )
            action = a.get("action_required", "")
            if action:
                parts.append(f"<p><strong>Action:</strong> {_esc(action)}</p>")
            parts.append("</div>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ascii_gauge(value: float, width: int = 20) -> str:
        """Create an ASCII gauge bar."""
        filled = int((value / 100) * width)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    @staticmethod
    def _ascii_bar(value: float, width: int = 40) -> str:
        """Create an ASCII bar."""
        filled = int((value / 100) * width)
        return f"[{'=' * filled}{' ' * (width - filled)}]"

    @staticmethod
    def _score_color(score: float) -> str:
        """Return color based on score."""
        if score >= 80:
            return "#2ecc71"
        if score >= 60:
            return "#27ae60"
        if score >= 40:
            return "#f39c12"
        return "#e74c3c"

    @staticmethod
    def _format_classification(classification: str) -> str:
        """Format SFDR classification."""
        mapping = {
            "article_8": "Article 8",
            "article_8_plus": "Article 8+",
            "article_9": "Article 9",
        }
        return mapping.get(classification, classification)

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
        """Wrap HTML in a complete document with dashboard-style CSS."""
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px auto; "
            "color: #2c3e50; line-height: 1.5; max-width: 1200px; background: #ecf0f1; }\n"
            "h1 { color: #1a5276; }\n"
            "h2 { color: #1a5276; margin-top: 0; font-size: 1.2em; }\n"
            ".card { background: white; border-radius: 8px; padding: 20px; "
            "margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n"
            ".card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); "
            "gap: 15px; margin: 10px 0; }\n"
            ".metric-box { padding: 12px; background: #f8f9fa; border-radius: 6px; text-align: center; }\n"
            ".metric-box .label { display: block; font-size: 0.85em; color: #7f8c8d; "
            "margin-bottom: 5px; }\n"
            ".metric-box .value { display: block; font-size: 1.3em; font-weight: bold; }\n"
            ".gauge-container { height: 8px; background: #ecf0f1; border-radius: 4px; "
            "margin: 8px 0; overflow: hidden; }\n"
            ".gauge-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }\n"
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
