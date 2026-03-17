"""
SustainableDashboardTemplate - Article 9 sustainable investment dashboard.

This module implements the sustainable investment dashboard template for
PACK-011 SFDR Article 9 products. It provides an interactive-style
dashboard with ASCII gauges showing sustainable proportion (~100%),
DNSH compliance rate, governance pass rate, taxonomy alignment,
PAI summary, impact highlights, downgrade risk indicators, and
carbon trajectory visualization.

Article 9 products must maintain near-100% sustainable investment
allocation (excluding cash/hedging), making real-time dashboard
monitoring essential for compliance and early-warning detection.

Example:
    >>> template = SustainableDashboardTemplate()
    >>> data = SustainableDashboardData(
    ...     fund_info=DashboardFundInfo(fund_name="Climate Impact Fund", ...),
    ...     sustainable_metrics=SustainableMetrics(...),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

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

class DashboardFundInfo(BaseModel):
    """Fund information for the dashboard."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    sfdr_classification: str = Field("article_9", description="SFDR classification")
    reporting_date: str = Field("", description="Dashboard date (YYYY-MM-DD)")
    currency: str = Field("EUR", description="Base currency")
    nav: Optional[float] = Field(None, ge=0.0, description="Current NAV")
    total_holdings: int = Field(0, ge=0, description="Number of holdings")
    management_company: str = Field("", description="Management company")
    benchmark_name: str = Field("", description="Designated benchmark")


class SustainableMetrics(BaseModel):
    """Core sustainable investment metrics for Article 9."""

    sustainable_pct: float = Field(
        100.0, ge=0.0, le=100.0,
        description="Total sustainable investment percentage (target ~100%)",
    )
    taxonomy_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="EU Taxonomy aligned percentage"
    )
    other_environmental_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Other environmental percentage"
    )
    social_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Social investment percentage"
    )
    cash_hedging_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Cash/hedging percentage"
    )
    minimum_commitment_pct: float = Field(
        95.0, ge=0.0, le=100.0,
        description="Minimum sustainable investment commitment",
    )


class DNSHMetrics(BaseModel):
    """Do No Significant Harm compliance metrics."""

    dnsh_pass_rate: float = Field(
        100.0, ge=0.0, le=100.0, description="DNSH pass rate (%)"
    )
    holdings_assessed: int = Field(
        0, ge=0, description="Number of holdings assessed"
    )
    holdings_passed: int = Field(
        0, ge=0, description="Number of holdings passing DNSH"
    )
    holdings_failed: int = Field(
        0, ge=0, description="Number of holdings failing DNSH"
    )
    failed_holdings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Failed holdings: {name, indicator, reason}",
    )
    pai_coverage_pct: float = Field(
        100.0, ge=0.0, le=100.0, description="PAI indicator coverage (%)"
    )


class GovernanceMetrics(BaseModel):
    """Good governance assessment metrics."""

    governance_pass_rate: float = Field(
        100.0, ge=0.0, le=100.0, description="Governance pass rate (%)"
    )
    holdings_assessed: int = Field(0, ge=0, description="Holdings assessed")
    holdings_passed: int = Field(0, ge=0, description="Holdings passing")
    ungc_compliant_pct: float = Field(
        100.0, ge=0.0, le=100.0, description="UN Global Compact compliance (%)"
    )
    governance_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Governance flags: {holding, flag, severity}",
    )


class TaxonomyMetrics(BaseModel):
    """EU Taxonomy alignment metrics."""

    total_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Total taxonomy aligned (%)"
    )
    committed_minimum_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Committed minimum alignment (%)"
    )
    eligible_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Taxonomy eligible (%)"
    )
    objective_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="By objective: {objective, aligned_pct}",
    )
    enabling_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Enabling activities (%)"
    )
    transitional_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Transitional activities (%)"
    )


class PAISummaryMetrics(BaseModel):
    """Principal Adverse Impact summary for dashboard."""

    total_indicators: int = Field(18, ge=0, description="Total PAI indicators")
    indicators_reported: int = Field(0, ge=0, description="Indicators reported")
    indicators_improved: int = Field(
        0, ge=0, description="Indicators improved YoY"
    )
    indicators_worsened: int = Field(
        0, ge=0, description="Indicators worsened YoY"
    )
    key_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key PAI metrics: {indicator, value, unit, trend}",
    )
    carbon_footprint: Optional[float] = Field(
        None, description="Portfolio carbon footprint (tCO2e/M invested)"
    )
    ghg_intensity: Optional[float] = Field(
        None, description="GHG intensity (tCO2e/M revenue)"
    )


class ImpactHighlights(BaseModel):
    """Impact highlights for the dashboard."""

    highlights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Impact highlights: {metric, value, unit, description, trend}",
    )
    sdg_contributions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top SDG contributions: {sdg_number, sdg_name, contribution_pct}",
    )
    engagement_outcomes: List[str] = Field(
        default_factory=list, description="Recent engagement outcomes"
    )


class DowngradeRisk(BaseModel):
    """Article 9 downgrade risk indicators."""

    overall_risk_level: str = Field(
        "low", description="Overall downgrade risk: low, medium, high"
    )
    risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Composite risk score (0-100)"
    )
    risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk factors: {factor, status, weight, description}",
    )
    sustainable_pct_trend: str = Field(
        "stable",
        description="Sustainable % trend: improving, stable, declining",
    )
    regulatory_changes: List[str] = Field(
        default_factory=list,
        description="Upcoming regulatory changes affecting classification",
    )
    remediation_actions: List[str] = Field(
        default_factory=list,
        description="Recommended remediation actions",
    )


class CarbonTrajectoryDashboard(BaseModel):
    """Carbon trajectory summary for dashboard."""

    current_intensity: Optional[float] = Field(
        None, description="Current carbon intensity"
    )
    target_intensity: Optional[float] = Field(
        None, description="Target carbon intensity"
    )
    baseline_intensity: Optional[float] = Field(
        None, description="Baseline carbon intensity"
    )
    on_track: bool = Field(True, description="Whether on track for target")
    annual_reduction_pct: Optional[float] = Field(
        None, description="Actual annual reduction (%)"
    )
    required_reduction_pct: float = Field(
        7.0, ge=0.0, description="Required annual reduction (%)"
    )
    years_to_target: Optional[int] = Field(
        None, ge=0, description="Estimated years to reach target"
    )
    net_zero_year: Optional[int] = Field(
        None, description="Projected net zero year"
    )


class SustainableDashboardData(BaseModel):
    """Complete input data for the sustainable dashboard."""

    fund_info: DashboardFundInfo
    sustainable_metrics: SustainableMetrics = Field(
        default_factory=SustainableMetrics
    )
    dnsh_metrics: DNSHMetrics = Field(default_factory=DNSHMetrics)
    governance_metrics: GovernanceMetrics = Field(
        default_factory=GovernanceMetrics
    )
    taxonomy_metrics: TaxonomyMetrics = Field(
        default_factory=TaxonomyMetrics
    )
    pai_summary: PAISummaryMetrics = Field(
        default_factory=PAISummaryMetrics
    )
    impact_highlights: ImpactHighlights = Field(
        default_factory=ImpactHighlights
    )
    downgrade_risk: DowngradeRisk = Field(default_factory=DowngradeRisk)
    carbon_trajectory: CarbonTrajectoryDashboard = Field(
        default_factory=CarbonTrajectoryDashboard
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class SustainableDashboardTemplate:
    """
    Sustainable investment dashboard template for Article 9 products.

    Provides an interactive-style compliance dashboard with ASCII gauges
    for sustainable proportion, DNSH rate, governance pass rate, taxonomy
    alignment, PAI summary, impact highlights, downgrade risk, and
    carbon trajectory visualization.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = SustainableDashboardTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Sustainable Proportion" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "sustainable_dashboard"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize SustainableDashboardTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    #  Public render dispatcher
    # ------------------------------------------------------------------ #

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render sustainable dashboard in the specified format.

        Args:
            data: Report data dictionary matching SustainableDashboardData schema.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered content as string (markdown/html) or dict (json).

        Raises:
            ValueError: If format is not supported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the sustainable dashboard as Markdown.

        Args:
            data: Report data dictionary matching SustainableDashboardData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_gauges(data))
        sections.append(self._md_section_2_sustainable_allocation(data))
        sections.append(self._md_section_3_dnsh(data))
        sections.append(self._md_section_4_governance(data))
        sections.append(self._md_section_5_taxonomy(data))
        sections.append(self._md_section_6_pai_summary(data))
        sections.append(self._md_section_7_impact(data))
        sections.append(self._md_section_8_downgrade_risk(data))
        sections.append(self._md_section_9_carbon_trajectory(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the sustainable dashboard as self-contained HTML.

        Args:
            data: Report data dictionary matching SustainableDashboardData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_gauges(data))
        sections.append(self._html_section_2_allocation(data))
        sections.append(self._html_section_3_pai(data))
        sections.append(self._html_section_4_impact(data))
        sections.append(self._html_section_5_downgrade(data))
        sections.append(self._html_section_6_carbon(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 Sustainable Investment Dashboard",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the sustainable dashboard as structured JSON.

        Args:
            data: Report data dictionary matching SustainableDashboardData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_sustainable_dashboard",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_info": data.get("fund_info", {}),
            "sustainable_metrics": data.get("sustainable_metrics", {}),
            "dnsh_metrics": data.get("dnsh_metrics", {}),
            "governance_metrics": data.get("governance_metrics", {}),
            "taxonomy_metrics": data.get("taxonomy_metrics", {}),
            "pai_summary": data.get("pai_summary", {}),
            "impact_highlights": data.get("impact_highlights", {}),
            "downgrade_risk": data.get("downgrade_risk", {}),
            "carbon_trajectory": data.get("carbon_trajectory", {}),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  ASCII Gauge Builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ascii_gauge(label: str, value: float, max_val: float = 100.0, width: int = 30) -> str:
        """Build an ASCII gauge bar for Markdown output."""
        ratio = min(value / max_val, 1.0) if max_val > 0 else 0.0
        filled = int(ratio * width)
        empty = width - filled
        bar = "#" * filled + "." * empty
        return f"  {label:28s} [{bar}] {value:.1f}%"

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        fi = data.get("fund_info", {})
        name = fi.get("fund_name", "Unknown Fund")
        return (
            f"# Sustainable Investment Dashboard (SFDR Article 9)\n\n"
            f"**Fund:** {name}\n\n"
            f"**ISIN:** {fi.get('isin', 'N/A') or 'N/A'} | "
            f"**Classification:** {fi.get('sfdr_classification', 'article_9').upper()} | "
            f"**NAV:** {fi.get('currency', 'EUR')} "
            f"{fi.get('nav', 0.0):,.0f}\n\n"
            f"**Dashboard Date:** {fi.get('reporting_date', '')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_gauges(self, data: Dict[str, Any]) -> str:
        """Section 1: Key compliance gauges."""
        sm = data.get("sustainable_metrics", {})
        dnsh = data.get("dnsh_metrics", {})
        gov = data.get("governance_metrics", {})
        tax = data.get("taxonomy_metrics", {})

        sustainable_pct = sm.get("sustainable_pct", 100.0)
        dnsh_rate = dnsh.get("dnsh_pass_rate", 100.0)
        gov_rate = gov.get("governance_pass_rate", 100.0)
        tax_aligned = tax.get("total_aligned_pct", 0.0)

        lines: List[str] = [
            "## Key Compliance Gauges\n",
            "```",
            self._ascii_gauge("Sustainable Proportion", sustainable_pct),
            self._ascii_gauge("DNSH Pass Rate", dnsh_rate),
            self._ascii_gauge("Governance Pass Rate", gov_rate),
            self._ascii_gauge("Taxonomy Alignment", tax_aligned),
            "```\n",
        ]

        # Status summary
        status_items = [
            ("Sustainable Proportion", sustainable_pct, 95.0),
            ("DNSH Rate", dnsh_rate, 100.0),
            ("Governance", gov_rate, 100.0),
        ]
        lines.append("| Metric | Value | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")
        for metric, val, threshold in status_items:
            status = "PASS" if val >= threshold else "WARN"
            lines.append(
                f"| {metric} | {val:.1f}% | {threshold:.0f}% | {status} |"
            )

        return "\n".join(lines)

    def _md_section_2_sustainable_allocation(self, data: Dict[str, Any]) -> str:
        """Section 2: Sustainable allocation breakdown."""
        sm = data.get("sustainable_metrics", {})
        sustainable = sm.get("sustainable_pct", 100.0)
        taxonomy = sm.get("taxonomy_aligned_pct", 0.0)
        other_env = sm.get("other_environmental_pct", 0.0)
        social = sm.get("social_pct", 0.0)
        cash = sm.get("cash_hedging_pct", 0.0)
        minimum = sm.get("minimum_commitment_pct", 95.0)

        lines: List[str] = [
            "## Sustainable Allocation\n",
            f"**Total Sustainable:** {sustainable:.1f}% | "
            f"**Minimum Commitment:** {minimum:.1f}%\n",
            "| Category | Allocation |",
            "|----------|-----------|",
            f"| Taxonomy-aligned | {taxonomy:.1f}% |",
            f"| Other environmental | {other_env:.1f}% |",
            f"| Social | {social:.1f}% |",
            f"| Cash/Hedging | {cash:.1f}% |",
            "",
            "### Allocation Chart\n",
            "```",
        ]

        items = [
            ("Taxonomy-aligned", taxonomy),
            ("Other environmental", other_env),
            ("Social", social),
            ("Cash/Hedging", cash),
        ]
        for label, pct in items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_3_dnsh(self, data: Dict[str, Any]) -> str:
        """Section 3: DNSH compliance."""
        dnsh = data.get("dnsh_metrics", {})
        rate = dnsh.get("dnsh_pass_rate", 100.0)
        assessed = dnsh.get("holdings_assessed", 0)
        passed = dnsh.get("holdings_passed", 0)
        failed = dnsh.get("holdings_failed", 0)
        failed_list = dnsh.get("failed_holdings", [])
        pai_cov = dnsh.get("pai_coverage_pct", 100.0)

        lines: List[str] = [
            "## DNSH Compliance\n",
            f"**Pass Rate:** {rate:.1f}% | **PAI Coverage:** {pai_cov:.1f}%\n",
            f"**Assessed:** {assessed} | **Passed:** {passed} | **Failed:** {failed}\n",
        ]

        if failed_list:
            lines.append("### Failed Holdings\n")
            lines.append("| Holding | Indicator | Reason |")
            lines.append("|---------|-----------|--------|")
            for h in failed_list:
                lines.append(
                    f"| {h.get('name', '')} | "
                    f"{h.get('indicator', '')} | "
                    f"{h.get('reason', '')} |"
                )

        return "\n".join(lines)

    def _md_section_4_governance(self, data: Dict[str, Any]) -> str:
        """Section 4: Governance assessment."""
        gov = data.get("governance_metrics", {})
        rate = gov.get("governance_pass_rate", 100.0)
        assessed = gov.get("holdings_assessed", 0)
        passed = gov.get("holdings_passed", 0)
        ungc = gov.get("ungc_compliant_pct", 100.0)
        flags = gov.get("governance_flags", [])

        lines: List[str] = [
            "## Governance Assessment\n",
            f"**Pass Rate:** {rate:.1f}% | **UNGC Compliance:** {ungc:.1f}%\n",
            f"**Assessed:** {assessed} | **Passed:** {passed}\n",
        ]

        if flags:
            lines.append("### Governance Flags\n")
            lines.append("| Holding | Flag | Severity |")
            lines.append("|---------|------|----------|")
            for f_item in flags:
                lines.append(
                    f"| {f_item.get('holding', '')} | "
                    f"{f_item.get('flag', '')} | "
                    f"{f_item.get('severity', '')} |"
                )

        return "\n".join(lines)

    def _md_section_5_taxonomy(self, data: Dict[str, Any]) -> str:
        """Section 5: Taxonomy alignment."""
        tax = data.get("taxonomy_metrics", {})
        aligned = tax.get("total_aligned_pct", 0.0)
        committed = tax.get("committed_minimum_pct", 0.0)
        eligible = tax.get("eligible_pct", 0.0)
        objectives = tax.get("objective_breakdown", [])
        enabling = tax.get("enabling_pct", 0.0)
        transitional = tax.get("transitional_pct", 0.0)

        lines: List[str] = [
            "## EU Taxonomy Alignment\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Aligned** | {aligned:.1f}% |",
            f"| **Committed Minimum** | {committed:.1f}% |",
            f"| **Eligible** | {eligible:.1f}% |",
            f"| **Enabling Activities** | {enabling:.1f}% |",
            f"| **Transitional Activities** | {transitional:.1f}% |",
            "",
        ]

        if objectives:
            lines.append("### By Environmental Objective\n")
            lines.append("| Objective | Aligned (%) |")
            lines.append("|-----------|------------|")
            for obj in objectives:
                lines.append(
                    f"| {obj.get('objective', '')} | "
                    f"{obj.get('aligned_pct', 0.0):.1f}% |"
                )

        return "\n".join(lines)

    def _md_section_6_pai_summary(self, data: Dict[str, Any]) -> str:
        """Section 6: PAI summary tiles."""
        pai = data.get("pai_summary", {})
        total = pai.get("total_indicators", 18)
        reported = pai.get("indicators_reported", 0)
        improved = pai.get("indicators_improved", 0)
        worsened = pai.get("indicators_worsened", 0)
        carbon_fp = pai.get("carbon_footprint")
        ghg_int = pai.get("ghg_intensity")
        key_metrics = pai.get("key_metrics", [])

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        lines: List[str] = [
            "## PAI Summary\n",
            f"**Reported:** {reported}/{total} | "
            f"**Improved:** {improved} | **Worsened:** {worsened}\n",
            f"**Carbon Footprint:** {_fmt(carbon_fp)} tCO2e/M invested | "
            f"**GHG Intensity:** {_fmt(ghg_int)} tCO2e/M revenue\n",
        ]

        if key_metrics:
            lines.append("### Key PAI Metrics\n")
            lines.append("| Indicator | Value | Unit | Trend |")
            lines.append("|-----------|-------|------|-------|")
            for m in key_metrics:
                trend_icon = {
                    "improving": "UP", "stable": "FLAT", "declining": "DOWN",
                }
                trend = m.get("trend", "stable")
                lines.append(
                    f"| {m.get('indicator', '')} | "
                    f"{m.get('value', '')} | "
                    f"{m.get('unit', '')} | "
                    f"{trend_icon.get(trend, trend)} |"
                )

        return "\n".join(lines)

    def _md_section_7_impact(self, data: Dict[str, Any]) -> str:
        """Section 7: Impact highlights."""
        imp = data.get("impact_highlights", {})
        highlights = imp.get("highlights", [])
        sdg_contribs = imp.get("sdg_contributions", [])
        engagements = imp.get("engagement_outcomes", [])

        lines: List[str] = [
            "## Impact Highlights\n",
        ]

        if highlights:
            lines.append("| Metric | Value | Unit | Trend |")
            lines.append("|--------|-------|------|-------|")
            for h in highlights:
                trend_icon = {
                    "improving": "UP", "stable": "FLAT", "declining": "DOWN",
                }
                trend = h.get("trend", "stable")
                lines.append(
                    f"| {h.get('metric', '')} | "
                    f"{h.get('value', '')} | "
                    f"{h.get('unit', '')} | "
                    f"{trend_icon.get(trend, trend)} |"
                )
            lines.append("")

        if sdg_contribs:
            lines.append("### Top SDG Contributions\n")
            for sdg in sdg_contribs:
                pct = sdg.get("contribution_pct", 0.0)
                bar_len = int(pct / 2)
                bar = "#" * bar_len
                lines.append(
                    f"- **SDG {sdg.get('sdg_number', '')} "
                    f"({sdg.get('sdg_name', '')}):** {pct:.1f}%"
                )
            lines.append("")

        if engagements:
            lines.append("### Recent Engagement Outcomes\n")
            for e in engagements:
                lines.append(f"- {e}")

        return "\n".join(lines)

    def _md_section_8_downgrade_risk(self, data: Dict[str, Any]) -> str:
        """Section 8: Downgrade risk assessment."""
        dr = data.get("downgrade_risk", {})
        level = dr.get("overall_risk_level", "low")
        score = dr.get("risk_score", 0.0)
        factors = dr.get("risk_factors", [])
        trend = dr.get("sustainable_pct_trend", "stable")
        changes = dr.get("regulatory_changes", [])
        actions = dr.get("remediation_actions", [])

        level_display = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}

        lines: List[str] = [
            "## Downgrade Risk Assessment\n",
            f"**Risk Level:** {level_display.get(level, level.upper())} | "
            f"**Risk Score:** {score:.1f}/100 | "
            f"**Sustainable % Trend:** {trend.upper()}\n",
        ]

        # ASCII gauge for risk
        lines.append("```")
        lines.append(self._ascii_gauge("Downgrade Risk", score))
        lines.append("```\n")

        if factors:
            lines.append("### Risk Factors\n")
            lines.append("| Factor | Status | Weight | Description |")
            lines.append("|--------|--------|--------|-------------|")
            for rf in factors:
                lines.append(
                    f"| {rf.get('factor', '')} | "
                    f"{rf.get('status', '')} | "
                    f"{rf.get('weight', '')} | "
                    f"{rf.get('description', '')} |"
                )
            lines.append("")

        if changes:
            lines.append("### Upcoming Regulatory Changes\n")
            for c in changes:
                lines.append(f"- {c}")
            lines.append("")

        if actions:
            lines.append("### Recommended Remediation Actions\n")
            for a in actions:
                lines.append(f"- {a}")

        return "\n".join(lines)

    def _md_section_9_carbon_trajectory(self, data: Dict[str, Any]) -> str:
        """Section 9: Carbon trajectory snapshot."""
        ct = data.get("carbon_trajectory", {})
        current = ct.get("current_intensity")
        target = ct.get("target_intensity")
        baseline = ct.get("baseline_intensity")
        on_track = ct.get("on_track", True)
        actual_red = ct.get("annual_reduction_pct")
        required_red = ct.get("required_reduction_pct", 7.0)
        years = ct.get("years_to_target")
        nz_year = ct.get("net_zero_year")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        lines: List[str] = [
            "## Carbon Trajectory\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Baseline Intensity** | {_fmt(baseline)} |",
            f"| **Current Intensity** | {_fmt(current)} |",
            f"| **Target Intensity** | {_fmt(target)} |",
            f"| **On Track** | {'Yes' if on_track else 'No'} |",
            f"| **Actual Annual Reduction** | {_fmt(actual_red)}% |",
            f"| **Required Annual Reduction** | {required_red:.1f}% |",
            f"| **Years to Target** | {years if years is not None else 'N/A'} |",
            f"| **Projected Net Zero** | {nz_year if nz_year is not None else 'N/A'} |",
            "",
        ]

        # ASCII trajectory visualization
        lines.append("### Trajectory Chart\n")
        lines.append("```")
        points = [
            ("Baseline", baseline or 0),
            ("Current", current or 0),
            ("Target", target or 0),
        ]
        max_val = max((v for _, v in points), default=1) or 1
        for label, val in points:
            bar_len = int((val / max_val) * 40)
            bar = "#" * bar_len
            lines.append(f"  {label:12s} [{bar:<40s}] {val:,.1f}")
        lines.append("```")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_gauges(self, data: Dict[str, Any]) -> str:
        """Build HTML gauges section."""
        sm = data.get("sustainable_metrics", {})
        dnsh = data.get("dnsh_metrics", {})
        gov = data.get("governance_metrics", {})
        tax = data.get("taxonomy_metrics", {})

        gauges = [
            ("Sustainable Proportion", sm.get("sustainable_pct", 100.0), "#27ae60"),
            ("DNSH Pass Rate", dnsh.get("dnsh_pass_rate", 100.0), "#2980b9"),
            ("Governance Pass Rate", gov.get("governance_pass_rate", 100.0), "#8e44ad"),
            ("Taxonomy Alignment", tax.get("total_aligned_pct", 0.0), "#f39c12"),
        ]

        parts: List[str] = [
            '<div class="section"><h2>Key Compliance Gauges</h2>',
            '<div style="display:flex;flex-wrap:wrap;gap:20px;">',
        ]

        for label, value, color in gauges:
            width_pct = min(value, 100.0)
            parts.append(
                f'<div style="flex:1;min-width:200px;text-align:center;">'
                f'<div style="font-weight:bold;margin-bottom:5px;">'
                f'{_esc(label)}</div>'
                f'<div style="background:#ecf0f1;border-radius:10px;height:20px;'
                f'overflow:hidden;">'
                f'<div style="background:{color};height:100%;width:{width_pct:.0f}%;'
                f'border-radius:10px;"></div></div>'
                f'<div style="margin-top:3px;font-size:1.2em;font-weight:bold;">'
                f'{value:.1f}%</div></div>'
            )

        parts.append("</div></div>")
        return "".join(parts)

    def _html_section_2_allocation(self, data: Dict[str, Any]) -> str:
        """Build HTML allocation section."""
        sm = data.get("sustainable_metrics", {})
        items = [
            ("Taxonomy-aligned", sm.get("taxonomy_aligned_pct", 0.0)),
            ("Other environmental", sm.get("other_environmental_pct", 0.0)),
            ("Social", sm.get("social_pct", 0.0)),
            ("Cash/Hedging", sm.get("cash_hedging_pct", 0.0)),
        ]

        parts: List[str] = [
            '<div class="section"><h2>Sustainable Allocation</h2>',
            '<table class="data-table">',
            "<tr><th>Category</th><th>Allocation</th></tr>",
        ]

        for label, pct in items:
            parts.append(
                f"<tr><td>{_esc(label)}</td><td>{pct:.1f}%</td></tr>"
            )
        parts.append("</table></div>")
        return "".join(parts)

    def _html_section_3_pai(self, data: Dict[str, Any]) -> str:
        """Build HTML PAI summary section."""
        pai = data.get("pai_summary", {})
        key_metrics = pai.get("key_metrics", [])

        parts: List[str] = [
            '<div class="section"><h2>PAI Summary</h2>',
            f'<p>Reported: {pai.get("indicators_reported", 0)}/'
            f'{pai.get("total_indicators", 18)} | '
            f'Improved: {pai.get("indicators_improved", 0)} | '
            f'Worsened: {pai.get("indicators_worsened", 0)}</p>',
        ]

        if key_metrics:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Indicator</th><th>Value</th><th>Unit</th><th>Trend</th></tr>")
            for m in key_metrics:
                trend = m.get("trend", "stable")
                trend_colors = {
                    "improving": "#27ae60", "stable": "#f39c12", "declining": "#e74c3c",
                }
                color = trend_colors.get(trend, "#7f8c8d")
                parts.append(
                    f"<tr><td>{_esc(str(m.get('indicator', '')))}</td>"
                    f"<td>{_esc(str(m.get('value', '')))}</td>"
                    f"<td>{_esc(str(m.get('unit', '')))}</td>"
                    f'<td style="color:{color};font-weight:bold;">'
                    f"{_esc(trend.upper())}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_impact(self, data: Dict[str, Any]) -> str:
        """Build HTML impact highlights section."""
        imp = data.get("impact_highlights", {})
        highlights = imp.get("highlights", [])

        parts: List[str] = [
            '<div class="section"><h2>Impact Highlights</h2>',
        ]

        if highlights:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Metric</th><th>Value</th><th>Unit</th><th>Trend</th></tr>")
            for h in highlights:
                parts.append(
                    f"<tr><td>{_esc(str(h.get('metric', '')))}</td>"
                    f"<td>{_esc(str(h.get('value', '')))}</td>"
                    f"<td>{_esc(str(h.get('unit', '')))}</td>"
                    f"<td>{_esc(str(h.get('trend', 'stable')).upper())}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_downgrade(self, data: Dict[str, Any]) -> str:
        """Build HTML downgrade risk section."""
        dr = data.get("downgrade_risk", {})
        level = dr.get("overall_risk_level", "low")
        score = dr.get("risk_score", 0.0)

        colors = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
        color = colors.get(level, "#7f8c8d")

        parts: List[str] = [
            '<div class="section"><h2>Downgrade Risk</h2>',
            f'<p>Risk Level: <span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:3px;font-weight:bold;">'
            f'{_esc(level.upper())}</span> | '
            f'Score: {score:.1f}/100</p>',
            "</div>",
        ]
        return "".join(parts)

    def _html_section_6_carbon(self, data: Dict[str, Any]) -> str:
        """Build HTML carbon trajectory section."""
        ct = data.get("carbon_trajectory", {})
        current = ct.get("current_intensity")
        target = ct.get("target_intensity")
        on_track = ct.get("on_track", True)

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        on_track_color = "#27ae60" if on_track else "#e74c3c"
        on_track_text = "On Track" if on_track else "Off Track"

        parts: List[str] = [
            '<div class="section"><h2>Carbon Trajectory</h2>',
            f'<p>Current: <strong>{_fmt(current)}</strong> | '
            f'Target: <strong>{_fmt(target)}</strong> | '
            f'<span style="color:{on_track_color};font-weight:bold;">'
            f'{on_track_text}</span></p>',
            "</div>",
        ]
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown footer with provenance."""
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; margin: 40px auto; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #1abc9c; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; "
            "padding-bottom: 5px; }\n"
            "h3 { color: #2c3e50; }\n"
            ".section { margin-bottom: 30px; padding: 15px; "
            "background: #fafafa; border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #1a5276; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID} | '
            f'{self.TEMPLATE_NAME} v{self.VERSION}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Module-level HTML escaping utility
# ---------------------------------------------------------------------------

def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
