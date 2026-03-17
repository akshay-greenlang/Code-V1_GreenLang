"""
PAIMandatoryReportTemplate - All 18 mandatory PAI indicators report.

This module implements the Principal Adverse Impact mandatory indicator
report template for PACK-011 SFDR Article 9 products. It covers all
18 mandatory PAI indicators from SFDR RTS Annex I Table 1, plus
additional optional indicators, integration in investment decisions,
actions taken, engagement targets, and data quality assessment.

Article 9 products must consider all mandatory PAI indicators and
demonstrate how adverse impacts are mitigated in pursuit of the
sustainable investment objective.

Example:
    >>> template = PAIMandatoryReportTemplate()
    >>> data = PAIMandatoryReportData(
    ...     entity_info=PAIEntityInfo(entity_name="Asset Management Co", ...),
    ...     mandatory_indicators=[...],
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

class PAIEntityInfo(BaseModel):
    """Entity information for PAI report."""

    entity_name: str = Field(..., min_length=1, description="Entity legal name")
    lei: str = Field("", description="Legal Entity Identifier")
    reporting_period_start: str = Field("", description="Period start (YYYY-MM-DD)")
    reporting_period_end: str = Field("", description="Period end (YYYY-MM-DD)")
    total_aum: Optional[float] = Field(
        None, ge=0.0, description="Total AUM in EUR"
    )
    number_of_products: int = Field(0, ge=0, description="Number of financial products")
    article_9_products: int = Field(0, ge=0, description="Number of Article 9 products")
    currency: str = Field("EUR", description="Reporting currency")


class MandatoryPAIIndicator(BaseModel):
    """A single mandatory PAI indicator (1-18)."""

    pai_number: int = Field(
        0, ge=1, le=18, description="PAI indicator number (1-18)"
    )
    indicator_name: str = Field("", description="Indicator name")
    metric: str = Field("", description="Metric description")
    category: str = Field(
        "",
        description="Category: climate_environment, social_governance, sovereigns, real_estate",
    )
    value_current: Optional[float] = Field(None, description="Current period value")
    value_previous: Optional[float] = Field(None, description="Previous period value")
    unit: str = Field("", description="Unit of measurement")
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Data coverage (%)"
    )
    explanation: str = Field("", description="Explanation of the indicator value")
    actions_taken: str = Field(
        "", description="Actions taken to address this indicator"
    )
    targets_set: str = Field(
        "", description="Targets set for this indicator"
    )
    data_source: str = Field("", description="Primary data source")
    estimation_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Percentage of estimated data"
    )


class AdditionalPAIIndicator(BaseModel):
    """An additional (optional) PAI indicator."""

    indicator_id: str = Field("", description="Indicator identifier (e.g., OPT-E01)")
    indicator_name: str = Field("", description="Indicator name")
    metric: str = Field("", description="Metric description")
    category: str = Field(
        "", description="Category: environment_additional, social_additional"
    )
    value_current: Optional[float] = Field(None, description="Current period value")
    value_previous: Optional[float] = Field(None, description="Previous period value")
    unit: str = Field("", description="Unit of measurement")
    rationale_for_selection: str = Field(
        "", description="Why this additional indicator was selected"
    )
    actions_taken: str = Field("", description="Actions taken")


class PAIIntegrationDescription(BaseModel):
    """Description of PAI integration in investment decisions."""

    policies_description: str = Field(
        "", description="Description of policies for identifying and prioritizing PAIs"
    )
    integration_approach: str = Field(
        "", description="How PAIs are integrated in the investment process"
    )
    screening_thresholds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Screening thresholds: {indicator, threshold, action}",
    )
    escalation_procedures: str = Field(
        "", description="Escalation procedures when thresholds are breached"
    )
    governance_structure: str = Field(
        "", description="Governance structure for PAI oversight"
    )


class PAIEngagement(BaseModel):
    """Engagement policies related to PAI indicators."""

    engagement_policy: str = Field(
        "", description="Summary of engagement policy"
    )
    engagement_activities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Engagement activities: {company, topic, outcome, date}",
    )
    voting_record_summary: str = Field(
        "", description="Summary of proxy voting record on ESG resolutions"
    )
    collaborative_initiatives: List[str] = Field(
        default_factory=list,
        description="Collaborative engagement initiatives participated in",
    )


class PAIDataQuality(BaseModel):
    """Data quality assessment for PAI reporting."""

    overall_quality_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall data quality score (%)"
    )
    reported_data_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Percentage of directly reported data"
    )
    estimated_data_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Percentage of estimated data"
    )
    data_providers: List[str] = Field(
        default_factory=list, description="External data providers used"
    )
    quality_issues: List[str] = Field(
        default_factory=list, description="Known data quality issues"
    )
    improvement_actions: List[str] = Field(
        default_factory=list, description="Planned data quality improvements"
    )


class PAIMandatoryReportData(BaseModel):
    """Complete input data for PAI mandatory report."""

    entity_info: PAIEntityInfo
    mandatory_indicators: List[MandatoryPAIIndicator] = Field(
        default_factory=list, description="All 18 mandatory PAI indicators"
    )
    additional_indicators: List[AdditionalPAIIndicator] = Field(
        default_factory=list, description="Selected additional indicators"
    )
    integration: PAIIntegrationDescription = Field(
        default_factory=PAIIntegrationDescription
    )
    engagement: PAIEngagement = Field(default_factory=PAIEngagement)
    data_quality: PAIDataQuality = Field(default_factory=PAIDataQuality)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class PAIMandatoryReportTemplate:
    """
    PAI mandatory indicator report template for Article 9 products.

    Generates a comprehensive PAI report covering all 18 mandatory
    indicators from SFDR RTS Annex I Table 1, additional indicators,
    integration in decisions, actions taken, targets, and data quality.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = PAIMandatoryReportTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "PAI 1" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "pai_mandatory_report"
    VERSION = "1.0"

    # Standard PAI categories for grouping
    PAI_CATEGORIES = {
        "climate_environment": "Climate and Environment-related (PAI 1-6)",
        "social_governance": "Social and Employee Matters (PAI 10-14)",
        "sovereigns": "Sovereigns and Supranationals (PAI 15-16)",
        "real_estate": "Real Estate Assets (PAI 17-18)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize PAIMandatoryReportTemplate.

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
        Render PAI mandatory report in the specified format.

        Args:
            data: Report data dictionary matching PAIMandatoryReportData schema.
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
        Render PAI mandatory report as Markdown.

        Args:
            data: Report data dictionary matching PAIMandatoryReportData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_summary(data))
        sections.append(self._md_section_2_mandatory_indicators(data))
        sections.append(self._md_section_3_additional_indicators(data))
        sections.append(self._md_section_4_integration(data))
        sections.append(self._md_section_5_engagement(data))
        sections.append(self._md_section_6_data_quality(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render PAI mandatory report as self-contained HTML.

        Args:
            data: Report data dictionary matching PAIMandatoryReportData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_summary(data))
        sections.append(self._html_section_2_indicators(data))
        sections.append(self._html_section_3_integration(data))
        sections.append(self._html_section_4_engagement(data))
        sections.append(self._html_section_5_data_quality(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 PAI Mandatory Indicators Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render PAI mandatory report as structured JSON.

        Args:
            data: Report data dictionary matching PAIMandatoryReportData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_pai_mandatory_report",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "entity_info": data.get("entity_info", {}),
            "mandatory_indicators": data.get("mandatory_indicators", []),
            "additional_indicators": data.get("additional_indicators", []),
            "integration": data.get("integration", {}),
            "engagement": data.get("engagement", {}),
            "data_quality": data.get("data_quality", {}),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        ei = data.get("entity_info", {})
        name = ei.get("entity_name", "Unknown Entity")
        start = ei.get("reporting_period_start", "")
        end = ei.get("reporting_period_end", "")
        return (
            f"# PAI Mandatory Indicators Report (SFDR Article 9)\n\n"
            f"**Entity:** {name}\n\n"
            f"**LEI:** {ei.get('lei', 'N/A') or 'N/A'}\n\n"
            f"**Reporting Period:** {start} to {end}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_summary(self, data: Dict[str, Any]) -> str:
        """Section 1: Summary overview."""
        ei = data.get("entity_info", {})
        indicators = data.get("mandatory_indicators", [])
        additional = data.get("additional_indicators", [])

        total_aum = ei.get("total_aum")
        aum_str = f"{total_aum:,.0f}" if total_aum is not None else "N/A"

        # Count indicators by trend
        improved = 0
        worsened = 0
        stable = 0
        for ind in indicators:
            curr = ind.get("value_current")
            prev = ind.get("value_previous")
            if curr is not None and prev is not None:
                if curr < prev:
                    improved += 1
                elif curr > prev:
                    worsened += 1
                else:
                    stable += 1

        lines: List[str] = [
            "## 1. Summary\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Total AUM** | {ei.get('currency', 'EUR')} {aum_str} |",
            f"| **Article 9 Products** | {ei.get('article_9_products', 0)} |",
            f"| **Mandatory Indicators Reported** | {len(indicators)}/18 |",
            f"| **Additional Indicators** | {len(additional)} |",
            f"| **Indicators Improved YoY** | {improved} |",
            f"| **Indicators Worsened YoY** | {worsened} |",
            f"| **Indicators Stable** | {stable} |",
        ]

        return "\n".join(lines)

    def _md_section_2_mandatory_indicators(self, data: Dict[str, Any]) -> str:
        """Section 2: All 18 mandatory PAI indicators."""
        indicators = data.get("mandatory_indicators", [])

        lines: List[str] = [
            "## 2. Mandatory PAI Indicators (1-18)\n",
        ]

        # Group by category
        categories = {
            "climate_environment": [],
            "social_governance": [],
            "sovereigns": [],
            "real_estate": [],
        }

        for ind in indicators:
            cat = ind.get("category", "climate_environment")
            if cat in categories:
                categories[cat].append(ind)
            else:
                categories["climate_environment"].append(ind)

        for cat_key, cat_label in self.PAI_CATEGORIES.items():
            cat_indicators = categories.get(cat_key, [])
            if not cat_indicators:
                continue

            lines.append(f"### {cat_label}\n")
            lines.append(
                "| # | Indicator | Current | Previous | Unit | Coverage | Trend |"
            )
            lines.append(
                "|---|-----------|---------|----------|------|----------|-------|"
            )

            for ind in cat_indicators:
                curr = ind.get("value_current")
                prev = ind.get("value_previous")
                curr_str = f"{curr:,.4f}" if curr is not None else "N/A"
                prev_str = f"{prev:,.4f}" if prev is not None else "N/A"
                coverage = ind.get("coverage_pct", 0.0)

                trend = "N/A"
                if curr is not None and prev is not None:
                    if curr < prev:
                        trend = "IMPROVED"
                    elif curr > prev:
                        trend = "WORSENED"
                    else:
                        trend = "STABLE"

                lines.append(
                    f"| PAI {ind.get('pai_number', '')} | "
                    f"{ind.get('indicator_name', '')} | "
                    f"{curr_str} | "
                    f"{prev_str} | "
                    f"{ind.get('unit', '')} | "
                    f"{coverage:.0f}% | "
                    f"{trend} |"
                )

            lines.append("")

            # Actions and targets per indicator
            for ind in cat_indicators:
                actions = ind.get("actions_taken", "")
                targets = ind.get("targets_set", "")
                explanation = ind.get("explanation", "")
                if actions or targets or explanation:
                    lines.append(
                        f"**PAI {ind.get('pai_number', '')} "
                        f"- {ind.get('indicator_name', '')}**\n"
                    )
                    if explanation:
                        lines.append(f"*Explanation:* {explanation}\n")
                    if actions:
                        lines.append(f"*Actions Taken:* {actions}\n")
                    if targets:
                        lines.append(f"*Targets:* {targets}\n")

        return "\n".join(lines)

    def _md_section_3_additional_indicators(self, data: Dict[str, Any]) -> str:
        """Section 3: Additional PAI indicators."""
        additional = data.get("additional_indicators", [])

        lines: List[str] = [
            "## 3. Additional PAI Indicators\n",
        ]

        if additional:
            lines.append("| ID | Indicator | Current | Previous | Unit | Rationale |")
            lines.append("|-----|-----------|---------|----------|------|-----------|")
            for ind in additional:
                curr = ind.get("value_current")
                prev = ind.get("value_previous")
                curr_str = f"{curr:,.4f}" if curr is not None else "N/A"
                prev_str = f"{prev:,.4f}" if prev is not None else "N/A"
                lines.append(
                    f"| {ind.get('indicator_id', '')} | "
                    f"{ind.get('indicator_name', '')} | "
                    f"{curr_str} | "
                    f"{prev_str} | "
                    f"{ind.get('unit', '')} | "
                    f"{ind.get('rationale_for_selection', '')} |"
                )
        else:
            lines.append("No additional indicators selected.")

        return "\n".join(lines)

    def _md_section_4_integration(self, data: Dict[str, Any]) -> str:
        """Section 4: Integration in investment decisions."""
        integ = data.get("integration", {})
        policies = integ.get("policies_description", "")
        approach = integ.get("integration_approach", "")
        thresholds = integ.get("screening_thresholds", [])
        escalation = integ.get("escalation_procedures", "")
        governance = integ.get("governance_structure", "")

        lines: List[str] = [
            "## 4. Integration in Investment Decisions\n",
        ]

        if policies:
            lines.append(f"### Policies\n\n{policies}\n")

        if approach:
            lines.append(f"### Integration Approach\n\n{approach}\n")

        if thresholds:
            lines.append("### Screening Thresholds\n")
            lines.append("| Indicator | Threshold | Action |")
            lines.append("|-----------|-----------|--------|")
            for t in thresholds:
                lines.append(
                    f"| {t.get('indicator', '')} | "
                    f"{t.get('threshold', '')} | "
                    f"{t.get('action', '')} |"
                )
            lines.append("")

        if escalation:
            lines.append(f"### Escalation Procedures\n\n{escalation}\n")

        if governance:
            lines.append(f"### Governance Structure\n\n{governance}")

        return "\n".join(lines)

    def _md_section_5_engagement(self, data: Dict[str, Any]) -> str:
        """Section 5: Engagement policies and outcomes."""
        eng = data.get("engagement", {})
        policy = eng.get("engagement_policy", "")
        activities = eng.get("engagement_activities", [])
        voting = eng.get("voting_record_summary", "")
        collaborative = eng.get("collaborative_initiatives", [])

        lines: List[str] = [
            "## 5. Engagement Policies\n",
        ]

        if policy:
            lines.append(f"### Policy Summary\n\n{policy}\n")

        if activities:
            lines.append("### Engagement Activities\n")
            lines.append("| Company | Topic | Outcome | Date |")
            lines.append("|---------|-------|---------|------|")
            for a in activities:
                lines.append(
                    f"| {a.get('company', '')} | "
                    f"{a.get('topic', '')} | "
                    f"{a.get('outcome', '')} | "
                    f"{a.get('date', '')} |"
                )
            lines.append("")

        if voting:
            lines.append(f"### Proxy Voting Record\n\n{voting}\n")

        if collaborative:
            lines.append("### Collaborative Initiatives\n")
            for c in collaborative:
                lines.append(f"- {c}")

        return "\n".join(lines)

    def _md_section_6_data_quality(self, data: Dict[str, Any]) -> str:
        """Section 6: Data quality assessment."""
        dq = data.get("data_quality", {})
        overall = dq.get("overall_quality_score", 0.0)
        reported = dq.get("reported_data_pct", 0.0)
        estimated = dq.get("estimated_data_pct", 0.0)
        providers = dq.get("data_providers", [])
        issues = dq.get("quality_issues", [])
        improvements = dq.get("improvement_actions", [])

        lines: List[str] = [
            "## 6. Data Quality\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Overall Quality Score** | {overall:.1f}% |",
            f"| **Reported Data** | {reported:.1f}% |",
            f"| **Estimated Data** | {estimated:.1f}% |",
            "",
        ]

        if providers:
            lines.append("### Data Providers\n")
            for p in providers:
                lines.append(f"- {p}")
            lines.append("")

        if issues:
            lines.append("### Known Quality Issues\n")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

        if improvements:
            lines.append("### Planned Improvements\n")
            for imp in improvements:
                lines.append(f"- {imp}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML summary section."""
        ei = data.get("entity_info", {})
        indicators = data.get("mandatory_indicators", [])

        parts: List[str] = [
            '<div class="section"><h2>1. Summary</h2>',
            f"<p><strong>{_esc(ei.get('entity_name', ''))}</strong> | "
            f"Article 9 Products: {ei.get('article_9_products', 0)} | "
            f"Indicators: {len(indicators)}/18</p>",
            "</div>",
        ]
        return "".join(parts)

    def _html_section_2_indicators(self, data: Dict[str, Any]) -> str:
        """Build HTML mandatory indicators section."""
        indicators = data.get("mandatory_indicators", [])

        parts: List[str] = [
            '<div class="section"><h2>2. Mandatory PAI Indicators</h2>',
            '<table class="data-table">',
            "<tr><th>#</th><th>Indicator</th><th>Current</th>"
            "<th>Previous</th><th>Unit</th><th>Coverage</th><th>Trend</th></tr>",
        ]

        for ind in indicators:
            curr = ind.get("value_current")
            prev = ind.get("value_previous")
            curr_str = f"{curr:,.4f}" if curr is not None else "N/A"
            prev_str = f"{prev:,.4f}" if prev is not None else "N/A"
            coverage = ind.get("coverage_pct", 0.0)

            trend = "N/A"
            trend_color = "#7f8c8d"
            if curr is not None and prev is not None:
                if curr < prev:
                    trend = "IMPROVED"
                    trend_color = "#27ae60"
                elif curr > prev:
                    trend = "WORSENED"
                    trend_color = "#e74c3c"
                else:
                    trend = "STABLE"
                    trend_color = "#f39c12"

            parts.append(
                f"<tr><td>PAI {ind.get('pai_number', '')}</td>"
                f"<td>{_esc(str(ind.get('indicator_name', '')))}</td>"
                f"<td>{curr_str}</td>"
                f"<td>{prev_str}</td>"
                f"<td>{_esc(str(ind.get('unit', '')))}</td>"
                f"<td>{coverage:.0f}%</td>"
                f'<td style="color:{trend_color};font-weight:bold;">{trend}</td></tr>'
            )

        parts.append("</table></div>")
        return "".join(parts)

    def _html_section_3_integration(self, data: Dict[str, Any]) -> str:
        """Build HTML integration section."""
        integ = data.get("integration", {})
        policies = integ.get("policies_description", "")
        approach = integ.get("integration_approach", "")

        parts: List[str] = [
            '<div class="section"><h2>4. Integration in Investment Decisions</h2>',
        ]

        if policies:
            parts.append(f"<h3>Policies</h3><p>{_esc(policies)}</p>")

        if approach:
            parts.append(f"<h3>Approach</h3><p>{_esc(approach)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_engagement(self, data: Dict[str, Any]) -> str:
        """Build HTML engagement section."""
        eng = data.get("engagement", {})
        activities = eng.get("engagement_activities", [])

        parts: List[str] = [
            '<div class="section"><h2>5. Engagement</h2>',
        ]

        if activities:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Company</th><th>Topic</th><th>Outcome</th><th>Date</th></tr>"
            )
            for a in activities:
                parts.append(
                    f"<tr><td>{_esc(str(a.get('company', '')))}</td>"
                    f"<td>{_esc(str(a.get('topic', '')))}</td>"
                    f"<td>{_esc(str(a.get('outcome', '')))}</td>"
                    f"<td>{_esc(str(a.get('date', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_data_quality(self, data: Dict[str, Any]) -> str:
        """Build HTML data quality section."""
        dq = data.get("data_quality", {})
        overall = dq.get("overall_quality_score", 0.0)
        reported = dq.get("reported_data_pct", 0.0)
        estimated = dq.get("estimated_data_pct", 0.0)

        bar_color = "#27ae60" if overall >= 80 else "#f39c12" if overall >= 60 else "#e74c3c"

        parts: List[str] = [
            '<div class="section"><h2>6. Data Quality</h2>',
            f'<p>Overall Score: <span style="font-size:1.3em;font-weight:bold;'
            f'color:{bar_color};">{overall:.1f}%</span> | '
            f"Reported: {reported:.1f}% | Estimated: {estimated:.1f}%</p>",
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
