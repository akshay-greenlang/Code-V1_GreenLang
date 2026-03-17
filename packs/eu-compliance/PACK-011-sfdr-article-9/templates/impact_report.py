"""
ImpactReportTemplate - Sustainable impact measurement report.

This module implements the impact report template for PACK-011
SFDR Article 9 products. It provides a comprehensive Theory of Change
framework, 15 environmental and 12 social KPIs, SDG mapping,
year-over-year comparison, and additionality assessment.

Article 9 products must demonstrate measurable impact toward their
sustainable investment objective, making this template a key compliance
and investor communication tool.

Example:
    >>> template = ImpactReportTemplate()
    >>> data = ImpactReportData(
    ...     fund_info=ImpactFundInfo(fund_name="Climate Impact Fund", ...),
    ...     theory_of_change=TheoryOfChange(...),
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

class ImpactFundInfo(BaseModel):
    """Fund information for impact report."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    reporting_period_start: str = Field("", description="Period start (YYYY-MM-DD)")
    reporting_period_end: str = Field("", description="Period end (YYYY-MM-DD)")
    currency: str = Field("EUR", description="Base currency")
    nav: Optional[float] = Field(None, ge=0.0, description="NAV at period end")
    total_holdings: int = Field(0, ge=0, description="Number of holdings")
    management_company: str = Field("", description="Management company")


class TheoryOfChange(BaseModel):
    """Theory of Change framework."""

    inputs: List[str] = Field(
        default_factory=list,
        description="Capital and resources deployed",
    )
    activities: List[str] = Field(
        default_factory=list,
        description="Investment activities and engagement",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Direct outputs from activities",
    )
    outcomes: List[str] = Field(
        default_factory=list,
        description="Short-to-medium term outcomes",
    )
    impacts: List[str] = Field(
        default_factory=list,
        description="Long-term systemic impacts",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions in the causal chain",
    )
    risks_to_impact: List[str] = Field(
        default_factory=list,
        description="Risks that could prevent impact realization",
    )


class EnvironmentalKPI(BaseModel):
    """Environmental KPI measurement."""

    kpi_id: str = Field("", description="KPI identifier (E01-E15)")
    name: str = Field("", description="KPI name")
    description: str = Field("", description="KPI description")
    value_current: Optional[float] = Field(None, description="Current period value")
    value_previous: Optional[float] = Field(None, description="Previous period value")
    target: Optional[float] = Field(None, description="Target value")
    unit: str = Field("", description="Unit of measurement")
    methodology: str = Field("", description="Measurement methodology")
    coverage_pct: float = Field(100.0, ge=0.0, le=100.0, description="Data coverage %")
    sdg_mapping: List[int] = Field(
        default_factory=list, description="Mapped SDG numbers"
    )


class SocialKPI(BaseModel):
    """Social KPI measurement."""

    kpi_id: str = Field("", description="KPI identifier (S01-S12)")
    name: str = Field("", description="KPI name")
    description: str = Field("", description="KPI description")
    value_current: Optional[float] = Field(None, description="Current period value")
    value_previous: Optional[float] = Field(None, description="Previous period value")
    target: Optional[float] = Field(None, description="Target value")
    unit: str = Field("", description="Unit of measurement")
    methodology: str = Field("", description="Measurement methodology")
    coverage_pct: float = Field(100.0, ge=0.0, le=100.0, description="Data coverage %")
    sdg_mapping: List[int] = Field(
        default_factory=list, description="Mapped SDG numbers"
    )


class SDGContribution(BaseModel):
    """SDG contribution mapping."""

    sdg_number: int = Field(0, ge=1, le=17, description="SDG number (1-17)")
    sdg_name: str = Field("", description="SDG name")
    contribution_level: str = Field(
        "primary",
        description="primary, secondary, or indirect",
    )
    percentage_aligned: float = Field(
        0.0, ge=0.0, le=100.0, description="% of portfolio aligned"
    )
    kpis_linked: List[str] = Field(
        default_factory=list, description="Linked KPI identifiers"
    )
    narrative: str = Field("", description="Contribution narrative")


class YoYComparison(BaseModel):
    """Year-over-year comparison data."""

    metric_name: str = Field("", description="Metric name")
    value_current: Optional[float] = Field(None, description="Current value")
    value_previous: Optional[float] = Field(None, description="Previous value")
    value_baseline: Optional[float] = Field(None, description="Baseline value")
    unit: str = Field("", description="Unit")
    trend: str = Field("stable", description="improving, stable, declining")
    commentary: str = Field("", description="Commentary on trend")


class AdditionalityAssessment(BaseModel):
    """Assessment of additionality of the fund's impact."""

    investor_contribution: str = Field(
        "", description="How investor capital contributes to impact"
    )
    counterfactual: str = Field(
        "",
        description="What would happen without this investment",
    )
    engagement_outcomes: List[str] = Field(
        default_factory=list,
        description="Outcomes from active engagement",
    )
    capital_allocation_impact: str = Field(
        "",
        description="Impact of capital allocation decisions",
    )
    signaling_effect: str = Field(
        "", description="Market signaling effects"
    )
    overall_assessment: str = Field(
        "moderate",
        description="high, moderate, low additionality assessment",
    )
    methodology_reference: str = Field(
        "",
        description="Reference to additionality methodology used",
    )


class ImpactReportData(BaseModel):
    """Complete input data for impact report."""

    fund_info: ImpactFundInfo
    theory_of_change: TheoryOfChange = Field(default_factory=TheoryOfChange)
    environmental_kpis: List[EnvironmentalKPI] = Field(
        default_factory=list, description="15 environmental KPIs"
    )
    social_kpis: List[SocialKPI] = Field(
        default_factory=list, description="12 social KPIs"
    )
    sdg_contributions: List[SDGContribution] = Field(
        default_factory=list, description="SDG contribution mapping"
    )
    yoy_comparisons: List[YoYComparison] = Field(
        default_factory=list, description="Year-over-year comparisons"
    )
    additionality: AdditionalityAssessment = Field(
        default_factory=AdditionalityAssessment
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class ImpactReportTemplate:
    """
    Sustainable impact measurement report template for Article 9 products.

    Provides comprehensive impact reporting including Theory of Change,
    environmental and social KPIs, SDG mapping, YoY trends, and
    additionality assessment.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = ImpactReportTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Theory of Change" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "impact_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ImpactReportTemplate.

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
        Render impact report in the specified format.

        Args:
            data: Report data dictionary matching ImpactReportData schema.
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
        Render the impact report as Markdown.

        Args:
            data: Report data dictionary matching ImpactReportData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_theory_of_change(data))
        sections.append(self._md_section_2_environmental_kpis(data))
        sections.append(self._md_section_3_social_kpis(data))
        sections.append(self._md_section_4_sdg_mapping(data))
        sections.append(self._md_section_5_yoy_comparison(data))
        sections.append(self._md_section_6_additionality(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the impact report as self-contained HTML.

        Args:
            data: Report data dictionary matching ImpactReportData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_theory_of_change(data))
        sections.append(self._html_section_2_environmental_kpis(data))
        sections.append(self._html_section_3_social_kpis(data))
        sections.append(self._html_section_4_sdg_mapping(data))
        sections.append(self._html_section_5_yoy_comparison(data))
        sections.append(self._html_section_6_additionality(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 Impact Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the impact report as structured JSON.

        Args:
            data: Report data dictionary matching ImpactReportData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        fi = data.get("fund_info", {})
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_impact_report",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_info": fi,
            "theory_of_change": data.get("theory_of_change", {}),
            "environmental_kpis": data.get("environmental_kpis", []),
            "social_kpis": data.get("social_kpis", []),
            "sdg_contributions": data.get("sdg_contributions", []),
            "yoy_comparisons": data.get("yoy_comparisons", []),
            "additionality": data.get("additionality", {}),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        fi = data.get("fund_info", {})
        name = fi.get("fund_name", "Unknown Fund")
        start = fi.get("reporting_period_start", "")
        end = fi.get("reporting_period_end", "")
        return (
            f"# Impact Report (SFDR Article 9)\n\n"
            f"**Fund:** {name}\n\n"
            f"**Reporting Period:** {start} to {end}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_theory_of_change(self, data: Dict[str, Any]) -> str:
        """Section 1: Theory of Change framework."""
        toc = data.get("theory_of_change", {})
        inputs = toc.get("inputs", [])
        activities = toc.get("activities", [])
        outputs = toc.get("outputs", [])
        outcomes = toc.get("outcomes", [])
        impacts = toc.get("impacts", [])
        assumptions = toc.get("assumptions", [])
        risks = toc.get("risks_to_impact", [])

        lines: List[str] = [
            "## 1. Theory of Change\n",
            "The following causal chain describes how capital deployment "
            "translates into measurable sustainable impact:\n",
        ]

        # ASCII flow diagram
        lines.append("```")
        lines.append("  INPUTS --> ACTIVITIES --> OUTPUTS --> OUTCOMES --> IMPACTS")
        lines.append("```\n")

        if inputs:
            lines.append("### Inputs (Capital & Resources)\n")
            for item in inputs:
                lines.append(f"- {item}")
            lines.append("")

        if activities:
            lines.append("### Activities\n")
            for item in activities:
                lines.append(f"- {item}")
            lines.append("")

        if outputs:
            lines.append("### Outputs\n")
            for item in outputs:
                lines.append(f"- {item}")
            lines.append("")

        if outcomes:
            lines.append("### Outcomes\n")
            for item in outcomes:
                lines.append(f"- {item}")
            lines.append("")

        if impacts:
            lines.append("### Long-Term Impacts\n")
            for item in impacts:
                lines.append(f"- {item}")
            lines.append("")

        if assumptions:
            lines.append("### Key Assumptions\n")
            for item in assumptions:
                lines.append(f"- {item}")
            lines.append("")

        if risks:
            lines.append("### Risks to Impact\n")
            for item in risks:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _md_section_2_environmental_kpis(self, data: Dict[str, Any]) -> str:
        """Section 2: Environmental KPIs (15 indicators)."""
        kpis = data.get("environmental_kpis", [])

        lines: List[str] = [
            "## 2. Environmental KPIs (E01-E15)\n",
        ]

        if kpis:
            lines.append("| ID | KPI | Current | Previous | Target | Unit | Coverage | SDGs |")
            lines.append("|-----|-----|---------|----------|--------|------|----------|------|")
            for kpi in kpis:
                current = kpi.get("value_current")
                previous = kpi.get("value_previous")
                target = kpi.get("target")
                curr_str = f"{current:,.2f}" if current is not None else "N/A"
                prev_str = f"{previous:,.2f}" if previous is not None else "N/A"
                target_str = f"{target:,.2f}" if target is not None else "-"
                sdgs = ", ".join(str(s) for s in kpi.get("sdg_mapping", []))
                lines.append(
                    f"| {kpi.get('kpi_id', '')} | "
                    f"{kpi.get('name', '')} | "
                    f"{curr_str} | "
                    f"{prev_str} | "
                    f"{target_str} | "
                    f"{kpi.get('unit', '')} | "
                    f"{kpi.get('coverage_pct', 100.0):.0f}% | "
                    f"{sdgs} |"
                )
        else:
            lines.append("No environmental KPI data available.")

        return "\n".join(lines)

    def _md_section_3_social_kpis(self, data: Dict[str, Any]) -> str:
        """Section 3: Social KPIs (12 indicators)."""
        kpis = data.get("social_kpis", [])

        lines: List[str] = [
            "## 3. Social KPIs (S01-S12)\n",
        ]

        if kpis:
            lines.append("| ID | KPI | Current | Previous | Target | Unit | Coverage | SDGs |")
            lines.append("|-----|-----|---------|----------|--------|------|----------|------|")
            for kpi in kpis:
                current = kpi.get("value_current")
                previous = kpi.get("value_previous")
                target = kpi.get("target")
                curr_str = f"{current:,.2f}" if current is not None else "N/A"
                prev_str = f"{previous:,.2f}" if previous is not None else "N/A"
                target_str = f"{target:,.2f}" if target is not None else "-"
                sdgs = ", ".join(str(s) for s in kpi.get("sdg_mapping", []))
                lines.append(
                    f"| {kpi.get('kpi_id', '')} | "
                    f"{kpi.get('name', '')} | "
                    f"{curr_str} | "
                    f"{prev_str} | "
                    f"{target_str} | "
                    f"{kpi.get('unit', '')} | "
                    f"{kpi.get('coverage_pct', 100.0):.0f}% | "
                    f"{sdgs} |"
                )
        else:
            lines.append("No social KPI data available.")

        return "\n".join(lines)

    def _md_section_4_sdg_mapping(self, data: Dict[str, Any]) -> str:
        """Section 4: SDG contribution mapping."""
        sdgs = data.get("sdg_contributions", [])

        lines: List[str] = [
            "## 4. SDG Contribution Mapping\n",
        ]

        if sdgs:
            lines.append("| SDG | Name | Level | Portfolio Aligned | Linked KPIs |")
            lines.append("|-----|------|-------|-------------------|-------------|")
            for sdg in sdgs:
                linked = ", ".join(sdg.get("kpis_linked", []))
                lines.append(
                    f"| SDG {sdg.get('sdg_number', '')} | "
                    f"{sdg.get('sdg_name', '')} | "
                    f"{sdg.get('contribution_level', '')} | "
                    f"{sdg.get('percentage_aligned', 0.0):.1f}% | "
                    f"{linked} |"
                )
            lines.append("")

            # Narratives
            for sdg in sdgs:
                narrative = sdg.get("narrative", "")
                if narrative:
                    lines.append(
                        f"**SDG {sdg.get('sdg_number', '')} "
                        f"- {sdg.get('sdg_name', '')}:** {narrative}\n"
                    )
        else:
            lines.append("No SDG contribution data available.")

        return "\n".join(lines)

    def _md_section_5_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Section 5: Year-over-year comparison."""
        comparisons = data.get("yoy_comparisons", [])

        lines: List[str] = [
            "## 5. Year-over-Year Comparison\n",
        ]

        if comparisons:
            lines.append("| Metric | Baseline | Previous | Current | Unit | Trend |")
            lines.append("|--------|----------|----------|---------|------|-------|")
            for comp in comparisons:
                baseline = comp.get("value_baseline")
                previous = comp.get("value_previous")
                current = comp.get("value_current")
                base_str = f"{baseline:,.2f}" if baseline is not None else "-"
                prev_str = f"{previous:,.2f}" if previous is not None else "-"
                curr_str = f"{current:,.2f}" if current is not None else "-"
                trend = comp.get("trend", "stable")
                trend_icon = {"improving": "UP", "stable": "FLAT", "declining": "DOWN"}
                lines.append(
                    f"| {comp.get('metric_name', '')} | "
                    f"{base_str} | "
                    f"{prev_str} | "
                    f"{curr_str} | "
                    f"{comp.get('unit', '')} | "
                    f"{trend_icon.get(trend, trend)} |"
                )
            lines.append("")

            # Commentaries
            for comp in comparisons:
                commentary = comp.get("commentary", "")
                if commentary:
                    lines.append(
                        f"**{comp.get('metric_name', '')}:** {commentary}\n"
                    )
        else:
            lines.append("No year-over-year comparison data available.")

        return "\n".join(lines)

    def _md_section_6_additionality(self, data: Dict[str, Any]) -> str:
        """Section 6: Additionality assessment."""
        add = data.get("additionality", {})
        investor_contribution = add.get("investor_contribution", "")
        counterfactual = add.get("counterfactual", "")
        outcomes = add.get("engagement_outcomes", [])
        capital_impact = add.get("capital_allocation_impact", "")
        signaling = add.get("signaling_effect", "")
        overall = add.get("overall_assessment", "moderate")
        methodology = add.get("methodology_reference", "")

        assessment_display = {
            "high": "HIGH",
            "moderate": "MODERATE",
            "low": "LOW",
        }

        lines: List[str] = [
            "## 6. Additionality Assessment\n",
            f"**Overall Assessment: {assessment_display.get(overall, overall.upper())}**\n",
        ]

        if investor_contribution:
            lines.append(f"### Investor Contribution\n\n{investor_contribution}\n")

        if counterfactual:
            lines.append(f"### Counterfactual Analysis\n\n{counterfactual}\n")

        if outcomes:
            lines.append("### Engagement Outcomes\n")
            for o in outcomes:
                lines.append(f"- {o}")
            lines.append("")

        if capital_impact:
            lines.append(f"### Capital Allocation Impact\n\n{capital_impact}\n")

        if signaling:
            lines.append(f"### Market Signaling Effect\n\n{signaling}\n")

        if methodology:
            lines.append(f"**Methodology Reference:** {methodology}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_theory_of_change(self, data: Dict[str, Any]) -> str:
        """Build HTML Theory of Change section."""
        toc = data.get("theory_of_change", {})

        parts: List[str] = [
            '<div class="section"><h2>1. Theory of Change</h2>',
            '<div style="text-align:center;padding:15px;background:#d5f5e3;'
            'border-radius:6px;margin:10px 0;font-weight:bold;">',
            "INPUTS &rarr; ACTIVITIES &rarr; OUTPUTS &rarr; OUTCOMES &rarr; IMPACTS",
            "</div>",
        ]

        stages = [
            ("Inputs", toc.get("inputs", [])),
            ("Activities", toc.get("activities", [])),
            ("Outputs", toc.get("outputs", [])),
            ("Outcomes", toc.get("outcomes", [])),
            ("Impacts", toc.get("impacts", [])),
        ]

        for stage_name, items in stages:
            if items:
                parts.append(f"<h3>{_esc(stage_name)}</h3><ul>")
                for item in items:
                    parts.append(f"<li>{_esc(item)}</li>")
                parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_environmental_kpis(self, data: Dict[str, Any]) -> str:
        """Build HTML environmental KPIs section."""
        kpis = data.get("environmental_kpis", [])
        parts: List[str] = [
            '<div class="section"><h2>2. Environmental KPIs (E01-E15)</h2>'
        ]

        if kpis:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>KPI</th><th>Current</th>"
                "<th>Previous</th><th>Target</th><th>Unit</th></tr>"
            )
            for kpi in kpis:
                current = kpi.get("value_current")
                previous = kpi.get("value_previous")
                target = kpi.get("target")
                curr_str = f"{current:,.2f}" if current is not None else "N/A"
                prev_str = f"{previous:,.2f}" if previous is not None else "N/A"
                tgt_str = f"{target:,.2f}" if target is not None else "-"
                parts.append(
                    f"<tr><td>{_esc(kpi.get('kpi_id', ''))}</td>"
                    f"<td>{_esc(kpi.get('name', ''))}</td>"
                    f"<td>{curr_str}</td>"
                    f"<td>{prev_str}</td>"
                    f"<td>{tgt_str}</td>"
                    f"<td>{_esc(kpi.get('unit', ''))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_social_kpis(self, data: Dict[str, Any]) -> str:
        """Build HTML social KPIs section."""
        kpis = data.get("social_kpis", [])
        parts: List[str] = [
            '<div class="section"><h2>3. Social KPIs (S01-S12)</h2>'
        ]

        if kpis:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>KPI</th><th>Current</th>"
                "<th>Previous</th><th>Target</th><th>Unit</th></tr>"
            )
            for kpi in kpis:
                current = kpi.get("value_current")
                previous = kpi.get("value_previous")
                target = kpi.get("target")
                curr_str = f"{current:,.2f}" if current is not None else "N/A"
                prev_str = f"{previous:,.2f}" if previous is not None else "N/A"
                tgt_str = f"{target:,.2f}" if target is not None else "-"
                parts.append(
                    f"<tr><td>{_esc(kpi.get('kpi_id', ''))}</td>"
                    f"<td>{_esc(kpi.get('name', ''))}</td>"
                    f"<td>{curr_str}</td>"
                    f"<td>{prev_str}</td>"
                    f"<td>{tgt_str}</td>"
                    f"<td>{_esc(kpi.get('unit', ''))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_sdg_mapping(self, data: Dict[str, Any]) -> str:
        """Build HTML SDG mapping section."""
        sdgs = data.get("sdg_contributions", [])
        parts: List[str] = [
            '<div class="section"><h2>4. SDG Contribution Mapping</h2>'
        ]

        if sdgs:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>SDG</th><th>Name</th><th>Level</th>"
                "<th>% Aligned</th><th>Linked KPIs</th></tr>"
            )
            for sdg in sdgs:
                linked = ", ".join(sdg.get("kpis_linked", []))
                parts.append(
                    f"<tr><td>SDG {sdg.get('sdg_number', '')}</td>"
                    f"<td>{_esc(str(sdg.get('sdg_name', '')))}</td>"
                    f"<td>{_esc(str(sdg.get('contribution_level', '')))}</td>"
                    f"<td>{sdg.get('percentage_aligned', 0.0):.1f}%</td>"
                    f"<td>{_esc(linked)}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML YoY comparison section."""
        comparisons = data.get("yoy_comparisons", [])
        parts: List[str] = [
            '<div class="section"><h2>5. Year-over-Year Comparison</h2>'
        ]

        if comparisons:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Metric</th><th>Previous</th><th>Current</th>"
                "<th>Unit</th><th>Trend</th></tr>"
            )
            for comp in comparisons:
                previous = comp.get("value_previous")
                current = comp.get("value_current")
                prev_str = f"{previous:,.2f}" if previous is not None else "-"
                curr_str = f"{current:,.2f}" if current is not None else "-"
                trend = comp.get("trend", "stable")
                trend_colors = {
                    "improving": "#27ae60",
                    "stable": "#f39c12",
                    "declining": "#e74c3c",
                }
                color = trend_colors.get(trend, "#7f8c8d")
                parts.append(
                    f"<tr><td>{_esc(str(comp.get('metric_name', '')))}</td>"
                    f"<td>{prev_str}</td>"
                    f"<td>{curr_str}</td>"
                    f"<td>{_esc(str(comp.get('unit', '')))}</td>"
                    f'<td style="color:{color};font-weight:bold;">'
                    f"{_esc(trend.upper())}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_additionality(self, data: Dict[str, Any]) -> str:
        """Build HTML additionality section."""
        add = data.get("additionality", {})
        overall = add.get("overall_assessment", "moderate")
        investor_contribution = add.get("investor_contribution", "")
        counterfactual = add.get("counterfactual", "")
        outcomes = add.get("engagement_outcomes", [])

        colors = {"high": "#27ae60", "moderate": "#f39c12", "low": "#e74c3c"}
        color = colors.get(overall, "#7f8c8d")

        parts: List[str] = [
            '<div class="section"><h2>6. Additionality Assessment</h2>',
            f'<p><span style="background:{color};color:white;padding:4px 12px;'
            f'border-radius:4px;font-weight:bold;">'
            f"{_esc(overall.upper())}</span></p>",
        ]

        if investor_contribution:
            parts.append(f"<h3>Investor Contribution</h3><p>{_esc(investor_contribution)}</p>")

        if counterfactual:
            parts.append(f"<h3>Counterfactual</h3><p>{_esc(counterfactual)}</p>")

        if outcomes:
            parts.append("<h3>Engagement Outcomes</h3><ul>")
            for o in outcomes:
                parts.append(f"<li>{_esc(o)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
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
