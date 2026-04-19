"""
AnnexVPeriodicTemplate - SFDR RTS Annex V periodic reporting template.

This module implements the Annex V periodic reporting template for
PACK-011 SFDR Article 9 products. It generates the mandatory periodic
disclosure required under SFDR Delegated Regulation (EU) 2022/1288,
Annex V for financial products that have sustainable investment as their
objective.

The template reports on: objective attainment, top investments, actual
proportions vs. commitments, taxonomy alignment bar charts, actions
taken during the period, mandatory PAI indicators, impact summary,
and benchmark performance comparison.

Example:
    >>> template = AnnexVPeriodicTemplate()
    >>> data = AnnexVPeriodicData(
    ...     reporting_period=Article9ReportingPeriod(
    ...         start_date="2025-01-01", end_date="2025-12-31",
    ...         fund_name="Climate Impact Fund"
    ...     ),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class Article9ReportingPeriod(BaseModel):
    """Reporting period identification for Article 9."""

    start_date: str = Field(..., description="Period start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Period end date (YYYY-MM-DD)")
    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    lei: str = Field("", description="Legal Entity Identifier")
    management_company: str = Field("", description="Management company name")
    sfdr_classification: str = Field("article_9", description="SFDR classification")
    fund_currency: str = Field("EUR", description="Fund base currency")
    nav_end_period: Optional[float] = Field(None, ge=0.0, description="NAV at period end")

    @field_validator("sfdr_classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        """Validate classification is Article 9."""
        allowed = {"article_9", "article_9_transitional"}
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v


class ObjectiveAttainment(BaseModel):
    """Attainment of sustainable investment objective during the period."""

    objective_description: str = Field("", description="Objective description")
    attainment_status: str = Field(
        "met", description="met, partially_met, not_met"
    )
    attainment_narrative: str = Field("", description="Narrative on attainment")
    kpi_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="KPI results: {name, target, actual, unit, status}",
    )
    impact_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Impact metrics: {metric, value, unit, yoy_change}",
    )


class PeriodicTopInvestment(BaseModel):
    """Top investment holding for periodic reporting."""

    rank: int = Field(0, ge=1, description="Rank position")
    name: str = Field("", description="Investment name")
    sector: str = Field("", description="Sector / NACE code")
    country: str = Field("", description="Country of domicile")
    weight_pct: float = Field(0.0, ge=0.0, le=100.0, description="Portfolio weight %")
    sustainable_objective: str = Field("", description="Sustainable objective contributed to")
    taxonomy_aligned: bool = Field(False, description="Taxonomy-aligned")


class ActualProportions(BaseModel):
    """Actual proportions achieved during the period."""

    sustainable_total_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Total sustainable %"
    )
    taxonomy_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Taxonomy-aligned %"
    )
    other_environmental_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Other environmental %"
    )
    social_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Social %"
    )
    not_sustainable_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Not sustainable (cash/hedging) %"
    )
    committed_sustainable_pct: float = Field(
        100.0, ge=0.0, le=100.0,
        description="Committed sustainable % (pre-contractual)",
    )
    committed_taxonomy_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Committed taxonomy-aligned % (pre-contractual)",
    )
    enabling_pct: float = Field(0.0, ge=0.0, le=100.0, description="Enabling %")
    transitional_pct: float = Field(0.0, ge=0.0, le=100.0, description="Transitional %")


class PeriodicPAIIndicator(BaseModel):
    """PAI indicator for periodic reporting."""

    pai_number: int = Field(0, ge=1, le=18, description="PAI indicator number")
    indicator_name: str = Field("", description="Indicator name")
    metric: str = Field("", description="Metric description")
    value_current: Optional[float] = Field(None, description="Current period value")
    value_previous: Optional[float] = Field(None, description="Previous period value")
    unit: str = Field("", description="Unit of measurement")
    explanation: str = Field("", description="Explanation of actions taken")


class ActionsTaken(BaseModel):
    """Actions taken during the reporting period."""

    engagement_activities: List[str] = Field(
        default_factory=list, description="Engagement activities undertaken"
    )
    voting_summary: str = Field("", description="Summary of proxy voting")
    divestments: List[str] = Field(
        default_factory=list, description="Divestment decisions made"
    )
    new_investments: List[str] = Field(
        default_factory=list, description="Key new sustainable investments"
    )
    policy_changes: List[str] = Field(
        default_factory=list, description="Investment policy changes"
    )


class ImpactSummary(BaseModel):
    """Summary of impact achieved during the period."""

    environmental_impacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Environmental impacts: {metric, value, unit, description}",
    )
    social_impacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Social impacts: {metric, value, unit, description}",
    )
    total_avoided_emissions: Optional[float] = Field(
        None, ge=0.0, description="Total avoided emissions (tCO2e)"
    )
    additionality_assessment: str = Field(
        "", description="Assessment of additionality"
    )


class BenchmarkPerformance(BaseModel):
    """Benchmark performance comparison."""

    has_benchmark: bool = Field(False, description="Whether benchmark is designated")
    benchmark_name: str = Field("", description="Benchmark name")
    benchmark_return_pct: Optional[float] = Field(None, description="Benchmark return %")
    fund_return_pct: Optional[float] = Field(None, description="Fund return %")
    tracking_error: Optional[float] = Field(None, ge=0.0, description="Tracking error")
    carbon_intensity_fund: Optional[float] = Field(None, ge=0.0, description="Fund WACI")
    carbon_intensity_benchmark: Optional[float] = Field(None, ge=0.0, description="Benchmark WACI")
    alignment_commentary: str = Field(
        "", description="Commentary on benchmark alignment"
    )


class AnnexVPeriodicData(BaseModel):
    """Complete input data for Annex V periodic disclosure (Article 9)."""

    reporting_period: Article9ReportingPeriod
    objective_attainment: ObjectiveAttainment = Field(
        default_factory=ObjectiveAttainment
    )
    top_investments: List[PeriodicTopInvestment] = Field(
        default_factory=list, description="Top 15 investments"
    )
    actual_proportions: ActualProportions = Field(
        default_factory=ActualProportions
    )
    pai_indicators: List[PeriodicPAIIndicator] = Field(
        default_factory=list, description="Mandatory PAI indicators"
    )
    actions_taken: ActionsTaken = Field(default_factory=ActionsTaken)
    impact_summary: ImpactSummary = Field(default_factory=ImpactSummary)
    benchmark_performance: BenchmarkPerformance = Field(
        default_factory=BenchmarkPerformance
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AnnexVPeriodicTemplate:
    """
    SFDR RTS Annex V periodic reporting template for Article 9 products.

    Generates the mandatory periodic disclosure required under SFDR
    Delegated Regulation (EU) 2022/1288 Annex V. Covers 8 sections
    for objective attainment, investments, proportions, taxonomy,
    PAI indicators, actions, impact, and benchmark performance.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = AnnexVPeriodicTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Objective Attainment" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "annex_v_periodic"
    VERSION = "1.0"

    SFDR_ARTICLE = "Article 9"
    REGULATION_REF = "Regulation (EU) 2019/2088"
    RTS_REF = "Delegated Regulation (EU) 2022/1288, Annex V"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AnnexVPeriodicTemplate.

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
        Render periodic disclosure in the specified format.

        Args:
            data: Report data dictionary matching AnnexVPeriodicData schema.
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
        Render the Annex V periodic disclosure as Markdown.

        Args:
            data: Report data dictionary matching AnnexVPeriodicData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_objective_attainment(data))
        sections.append(self._md_section_2_top_investments(data))
        sections.append(self._md_section_3_actual_proportions(data))
        sections.append(self._md_section_4_taxonomy_alignment(data))
        sections.append(self._md_section_5_pai_indicators(data))
        sections.append(self._md_section_6_actions_taken(data))
        sections.append(self._md_section_7_impact_summary(data))
        sections.append(self._md_section_8_benchmark_performance(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the Annex V periodic disclosure as self-contained HTML.

        Args:
            data: Report data dictionary matching AnnexVPeriodicData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_objective_attainment(data))
        sections.append(self._html_section_2_top_investments(data))
        sections.append(self._html_section_3_actual_proportions(data))
        sections.append(self._html_section_4_taxonomy_alignment(data))
        sections.append(self._html_section_5_pai_indicators(data))
        sections.append(self._html_section_6_actions_taken(data))
        sections.append(self._html_section_7_impact_summary(data))
        sections.append(self._html_section_8_benchmark_performance(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Annex V Periodic Report (Article 9)",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the Annex V periodic disclosure as structured JSON.

        Args:
            data: Report data dictionary matching AnnexVPeriodicData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        rp = data.get("reporting_period", {})
        report: Dict[str, Any] = {
            "report_type": "sfdr_annex_v_periodic",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "regulation": self.REGULATION_REF,
            "rts_annex": self.RTS_REF,
            "sfdr_article": self.SFDR_ARTICLE,
            "reporting_period": {
                "start_date": rp.get("start_date", ""),
                "end_date": rp.get("end_date", ""),
                "fund_name": rp.get("fund_name", ""),
                "isin": rp.get("isin", ""),
                "lei": rp.get("lei", ""),
                "nav_end_period": rp.get("nav_end_period"),
            },
            "objective_attainment": data.get("objective_attainment", {}),
            "top_investments": data.get("top_investments", []),
            "actual_proportions": data.get("actual_proportions", {}),
            "pai_indicators": data.get("pai_indicators", []),
            "actions_taken": data.get("actions_taken", {}),
            "impact_summary": data.get("impact_summary", {}),
            "benchmark_performance": data.get("benchmark_performance", {}),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        rp = data.get("reporting_period", {})
        name = rp.get("fund_name", "Unknown Fund")
        start = rp.get("start_date", "")
        end = rp.get("end_date", "")
        return (
            f"# Periodic Report ({self.SFDR_ARTICLE})\n\n"
            f"**{self.RTS_REF}**\n\n"
            f"> **This financial product had sustainable investment as its objective.**\n\n"
            f"**Fund:** {name}\n\n"
            f"**Reporting Period:** {start} to {end}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_objective_attainment(self, data: Dict[str, Any]) -> str:
        """Section 1: Sustainable objective attainment."""
        oa = data.get("objective_attainment", {})
        status = oa.get("attainment_status", "met")
        narrative = oa.get("attainment_narrative", "")
        kpis = oa.get("kpi_results", [])
        impact = oa.get("impact_metrics", [])
        obj_desc = oa.get("objective_description", "")

        status_display = {"met": "MET", "partially_met": "PARTIALLY MET", "not_met": "NOT MET"}

        lines: List[str] = [
            "## 1. Sustainable Investment Objective Attainment\n",
            f"**Status: {status_display.get(status, status.upper())}**\n",
        ]

        if obj_desc:
            lines.append(f"**Objective:** {obj_desc}\n")

        if narrative:
            lines.append(f"{narrative}\n")

        if kpis:
            lines.append("### KPI Results\n")
            lines.append("| KPI | Target | Actual | Unit | Status |")
            lines.append("|-----|--------|--------|------|--------|")
            for kpi in kpis:
                lines.append(
                    f"| {kpi.get('name', '')} | "
                    f"{kpi.get('target', '')} | "
                    f"{kpi.get('actual', '')} | "
                    f"{kpi.get('unit', '')} | "
                    f"{kpi.get('status', '')} |"
                )
            lines.append("")

        if impact:
            lines.append("### Impact Metrics\n")
            lines.append("| Metric | Value | Unit | YoY Change |")
            lines.append("|--------|-------|------|------------|")
            for m in impact:
                yoy = m.get("yoy_change", "")
                yoy_str = f"{yoy}" if yoy != "" else "N/A"
                lines.append(
                    f"| {m.get('metric', '')} | "
                    f"{m.get('value', '')} | "
                    f"{m.get('unit', '')} | "
                    f"{yoy_str} |"
                )

        return "\n".join(lines)

    def _md_section_2_top_investments(self, data: Dict[str, Any]) -> str:
        """Section 2: Top investments during the period."""
        investments = data.get("top_investments", [])

        lines: List[str] = [
            "## 2. Top Investments\n",
        ]

        if investments:
            lines.append("| # | Investment | Sector | Country | Weight | Sust. Obj. | Tax. Aligned |")
            lines.append("|---|-----------|--------|---------|--------|------------|--------------|")
            for inv in investments:
                tax_icon = "Yes" if inv.get("taxonomy_aligned", False) else "No"
                lines.append(
                    f"| {inv.get('rank', '')} | "
                    f"{inv.get('name', '')} | "
                    f"{inv.get('sector', '')} | "
                    f"{inv.get('country', '')} | "
                    f"{inv.get('weight_pct', 0.0):.1f}% | "
                    f"{inv.get('sustainable_objective', '')} | "
                    f"{tax_icon} |"
                )
        else:
            lines.append("No top investments data available.")

        return "\n".join(lines)

    def _md_section_3_actual_proportions(self, data: Dict[str, Any]) -> str:
        """Section 3: Actual proportions vs. commitments."""
        ap = data.get("actual_proportions", {})
        sustainable = ap.get("sustainable_total_pct", 0.0)
        taxonomy = ap.get("taxonomy_aligned_pct", 0.0)
        other_env = ap.get("other_environmental_pct", 0.0)
        social = ap.get("social_pct", 0.0)
        not_sust = ap.get("not_sustainable_pct", 0.0)
        committed_sust = ap.get("committed_sustainable_pct", 100.0)
        committed_tax = ap.get("committed_taxonomy_pct", 0.0)

        lines: List[str] = [
            "## 3. Actual Proportions of Sustainable Investments\n",
            "| Category | Committed | Actual | Delta |",
            "|----------|-----------|--------|-------|",
            f"| Sustainable Total | {committed_sust:.1f}% | {sustainable:.1f}% | "
            f"{sustainable - committed_sust:+.1f}pp |",
            f"| Taxonomy-aligned | {committed_tax:.1f}% | {taxonomy:.1f}% | "
            f"{taxonomy - committed_tax:+.1f}pp |",
            f"| Other environmental | - | {other_env:.1f}% | - |",
            f"| Social | - | {social:.1f}% | - |",
            f"| Not sustainable | - | {not_sust:.1f}% | - |",
            "",
        ]

        # ASCII bar comparison
        lines.append("### Committed vs. Actual\n")
        lines.append("```")
        lines.append(f"  Committed  [{('#' * int(committed_sust / 2)):<50s}] {committed_sust:.1f}%")
        lines.append(f"  Actual     [{('#' * int(sustainable / 2)):<50s}] {sustainable:.1f}%")
        lines.append("")
        chart_items = [
            ("Tax. aligned", taxonomy),
            ("Other environ.", other_env),
            ("Social", social),
            ("Cash/Hedging", not_sust),
        ]
        for label, pct in chart_items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_4_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Section 4: Taxonomy alignment actual results."""
        ap = data.get("actual_proportions", {})
        taxonomy = ap.get("taxonomy_aligned_pct", 0.0)
        enabling = ap.get("enabling_pct", 0.0)
        transitional = ap.get("transitional_pct", 0.0)

        lines: List[str] = [
            "## 4. EU Taxonomy Alignment\n",
            f"**Actual taxonomy-aligned proportion:** {taxonomy:.1f}%\n",
            f"**Enabling activities:** {enabling:.1f}%\n",
            f"**Transitional activities:** {transitional:.1f}%\n",
        ]

        # Bar chart
        lines.append("### Taxonomy Breakdown\n")
        lines.append("```")
        chart_items = [
            ("Taxonomy-aligned", taxonomy),
            ("Enabling", enabling),
            ("Transitional", transitional),
        ]
        for label, pct in chart_items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_5_pai_indicators(self, data: Dict[str, Any]) -> str:
        """Section 5: Mandatory PAI indicators."""
        indicators = data.get("pai_indicators", [])

        lines: List[str] = [
            "## 5. Mandatory Principal Adverse Impact Indicators\n",
        ]

        if indicators:
            lines.append("| PAI # | Indicator | Current | Previous | Unit | Explanation |")
            lines.append("|-------|-----------|---------|----------|------|-------------|")
            for ind in indicators:
                current = ind.get("value_current")
                previous = ind.get("value_previous")
                curr_str = f"{current:.4f}" if current is not None else "N/A"
                prev_str = f"{previous:.4f}" if previous is not None else "N/A"
                lines.append(
                    f"| {ind.get('pai_number', '')} | "
                    f"{ind.get('indicator_name', '')} | "
                    f"{curr_str} | "
                    f"{prev_str} | "
                    f"{ind.get('unit', '')} | "
                    f"{ind.get('explanation', '')} |"
                )
        else:
            lines.append("PAI indicator data not available for this period.")

        return "\n".join(lines)

    def _md_section_6_actions_taken(self, data: Dict[str, Any]) -> str:
        """Section 6: Actions taken during the period."""
        actions = data.get("actions_taken", {})
        engagement = actions.get("engagement_activities", [])
        voting = actions.get("voting_summary", "")
        divestments = actions.get("divestments", [])
        new_inv = actions.get("new_investments", [])
        policy = actions.get("policy_changes", [])

        lines: List[str] = [
            "## 6. Actions Taken\n",
        ]

        if engagement:
            lines.append("### Engagement Activities\n")
            for e in engagement:
                lines.append(f"- {e}")
            lines.append("")

        if voting:
            lines.append(f"### Proxy Voting\n\n{voting}\n")

        if divestments:
            lines.append("### Divestments\n")
            for d in divestments:
                lines.append(f"- {d}")
            lines.append("")

        if new_inv:
            lines.append("### Key New Investments\n")
            for n in new_inv:
                lines.append(f"- {n}")
            lines.append("")

        if policy:
            lines.append("### Policy Changes\n")
            for p in policy:
                lines.append(f"- {p}")

        if not any([engagement, voting, divestments, new_inv, policy]):
            lines.append("No significant actions to report for this period.")

        return "\n".join(lines)

    def _md_section_7_impact_summary(self, data: Dict[str, Any]) -> str:
        """Section 7: Impact summary."""
        impact = data.get("impact_summary", {})
        env_impacts = impact.get("environmental_impacts", [])
        soc_impacts = impact.get("social_impacts", [])
        avoided = impact.get("total_avoided_emissions")
        additionality = impact.get("additionality_assessment", "")

        lines: List[str] = [
            "## 7. Impact Summary\n",
        ]

        if avoided is not None:
            lines.append(f"**Total Avoided Emissions:** {avoided:,.1f} tCO2e\n")

        if env_impacts:
            lines.append("### Environmental Impacts\n")
            lines.append("| Metric | Value | Unit | Description |")
            lines.append("|--------|-------|------|-------------|")
            for ei in env_impacts:
                lines.append(
                    f"| {ei.get('metric', '')} | "
                    f"{ei.get('value', '')} | "
                    f"{ei.get('unit', '')} | "
                    f"{ei.get('description', '')} |"
                )
            lines.append("")

        if soc_impacts:
            lines.append("### Social Impacts\n")
            lines.append("| Metric | Value | Unit | Description |")
            lines.append("|--------|-------|------|-------------|")
            for si in soc_impacts:
                lines.append(
                    f"| {si.get('metric', '')} | "
                    f"{si.get('value', '')} | "
                    f"{si.get('unit', '')} | "
                    f"{si.get('description', '')} |"
                )
            lines.append("")

        if additionality:
            lines.append(f"### Additionality Assessment\n\n{additionality}")

        return "\n".join(lines)

    def _md_section_8_benchmark_performance(self, data: Dict[str, Any]) -> str:
        """Section 8: Benchmark performance comparison."""
        bm = data.get("benchmark_performance", {})
        has_bm = bm.get("has_benchmark", False)
        bm_name = bm.get("benchmark_name", "")
        bm_return = bm.get("benchmark_return_pct")
        fund_return = bm.get("fund_return_pct")
        tracking = bm.get("tracking_error")
        waci_fund = bm.get("carbon_intensity_fund")
        waci_bm = bm.get("carbon_intensity_benchmark")
        commentary = bm.get("alignment_commentary", "")

        lines: List[str] = [
            "## 8. Benchmark Performance\n",
        ]

        if has_bm:
            lines.append(f"**Designated Benchmark:** {bm_name}\n")
            lines.append("| Metric | Fund | Benchmark |")
            lines.append("|--------|------|-----------|")
            if fund_return is not None and bm_return is not None:
                lines.append(
                    f"| Return | {fund_return:.2f}% | {bm_return:.2f}% |"
                )
            if waci_fund is not None and waci_bm is not None:
                lines.append(
                    f"| WACI (tCO2e/EUR M) | {waci_fund:.1f} | {waci_bm:.1f} |"
                )
            if tracking is not None:
                lines.append(f"| Tracking Error | {tracking:.2f}% | - |")
            lines.append("")

            if commentary:
                lines.append(f"{commentary}")
        else:
            lines.append("No EU Climate Benchmark designated.\n")
            if commentary:
                lines.append(f"{commentary}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_objective_attainment(self, data: Dict[str, Any]) -> str:
        """Build HTML objective attainment section."""
        oa = data.get("objective_attainment", {})
        status = oa.get("attainment_status", "met")
        narrative = oa.get("attainment_narrative", "")
        kpis = oa.get("kpi_results", [])

        status_colors = {"met": "#27ae60", "partially_met": "#f39c12", "not_met": "#e74c3c"}
        status_labels = {"met": "MET", "partially_met": "PARTIALLY MET", "not_met": "NOT MET"}
        color = status_colors.get(status, "#7f8c8d")
        label = status_labels.get(status, status.upper())

        parts: List[str] = [
            '<div class="section">',
            "<h2>1. Sustainable Investment Objective Attainment</h2>",
            f'<p><span style="background:{color};color:white;padding:4px 12px;'
            f'border-radius:4px;font-weight:bold;">{_esc(label)}</span></p>',
        ]

        if narrative:
            parts.append(f"<p>{_esc(narrative)}</p>")

        if kpis:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>KPI</th><th>Target</th><th>Actual</th>"
                         "<th>Unit</th><th>Status</th></tr>")
            for kpi in kpis:
                parts.append(
                    f"<tr><td>{_esc(str(kpi.get('name', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('target', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('actual', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('unit', '')))}</td>"
                    f"<td>{_esc(str(kpi.get('status', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_top_investments(self, data: Dict[str, Any]) -> str:
        """Build HTML top investments section."""
        investments = data.get("top_investments", [])
        parts: List[str] = ['<div class="section"><h2>2. Top Investments</h2>']

        if investments:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>#</th><th>Investment</th><th>Sector</th>"
                "<th>Country</th><th>Weight</th><th>Tax. Aligned</th></tr>"
            )
            for inv in investments:
                tax = "Yes" if inv.get("taxonomy_aligned", False) else "No"
                parts.append(
                    f"<tr><td>{inv.get('rank', '')}</td>"
                    f"<td>{_esc(str(inv.get('name', '')))}</td>"
                    f"<td>{_esc(str(inv.get('sector', '')))}</td>"
                    f"<td>{_esc(str(inv.get('country', '')))}</td>"
                    f"<td>{inv.get('weight_pct', 0.0):.1f}%</td>"
                    f"<td>{tax}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_actual_proportions(self, data: Dict[str, Any]) -> str:
        """Build HTML actual proportions section."""
        ap = data.get("actual_proportions", {})

        parts: List[str] = [
            '<div class="section"><h2>3. Actual Proportions</h2>',
            '<table class="data-table">',
            "<tr><th>Category</th><th>Proportion</th><th>Visual</th></tr>",
        ]

        items = [
            ("Taxonomy-aligned", ap.get("taxonomy_aligned_pct", 0.0), "#2ecc71"),
            ("Other environmental", ap.get("other_environmental_pct", 0.0), "#27ae60"),
            ("Social", ap.get("social_pct", 0.0), "#3498db"),
            ("Cash/Hedging", ap.get("not_sustainable_pct", 0.0), "#95a5a6"),
        ]

        for label, pct, color in items:
            bar_width = max(int(pct * 2), 0)
            parts.append(
                f"<tr><td>{_esc(label)}</td><td>{pct:.1f}%</td>"
                f'<td><div style="background:{color};width:{bar_width}px;'
                f'height:16px;border-radius:3px;"></div></td></tr>'
            )

        parts.append("</table></div>")
        return "".join(parts)

    def _html_section_4_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        """Build HTML taxonomy alignment section."""
        ap = data.get("actual_proportions", {})
        taxonomy = ap.get("taxonomy_aligned_pct", 0.0)
        enabling = ap.get("enabling_pct", 0.0)
        transitional = ap.get("transitional_pct", 0.0)

        parts: List[str] = [
            '<div class="section"><h2>4. EU Taxonomy Alignment</h2>',
            f"<p><strong>Actual taxonomy-aligned:</strong> {taxonomy:.1f}%</p>",
            f"<p>Enabling: {enabling:.1f}% | Transitional: {transitional:.1f}%</p>",
            "</div>",
        ]
        return "".join(parts)

    def _html_section_5_pai_indicators(self, data: Dict[str, Any]) -> str:
        """Build HTML PAI indicators section."""
        indicators = data.get("pai_indicators", [])
        parts: List[str] = [
            '<div class="section"><h2>5. Mandatory PAI Indicators</h2>'
        ]

        if indicators:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>PAI #</th><th>Indicator</th><th>Current</th>"
                "<th>Previous</th><th>Unit</th></tr>"
            )
            for ind in indicators:
                current = ind.get("value_current")
                previous = ind.get("value_previous")
                curr_str = f"{current:.4f}" if current is not None else "N/A"
                prev_str = f"{previous:.4f}" if previous is not None else "N/A"
                parts.append(
                    f"<tr><td>{ind.get('pai_number', '')}</td>"
                    f"<td>{_esc(str(ind.get('indicator_name', '')))}</td>"
                    f"<td>{curr_str}</td>"
                    f"<td>{prev_str}</td>"
                    f"<td>{_esc(str(ind.get('unit', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_actions_taken(self, data: Dict[str, Any]) -> str:
        """Build HTML actions taken section."""
        actions = data.get("actions_taken", {})
        engagement = actions.get("engagement_activities", [])
        divestments = actions.get("divestments", [])

        parts: List[str] = [
            '<div class="section"><h2>6. Actions Taken</h2>'
        ]

        if engagement:
            parts.append("<h3>Engagement Activities</h3><ul>")
            for e in engagement:
                parts.append(f"<li>{_esc(e)}</li>")
            parts.append("</ul>")

        if divestments:
            parts.append("<h3>Divestments</h3><ul>")
            for d in divestments:
                parts.append(f"<li>{_esc(d)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_7_impact_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML impact summary section."""
        impact = data.get("impact_summary", {})
        env_impacts = impact.get("environmental_impacts", [])
        avoided = impact.get("total_avoided_emissions")

        parts: List[str] = [
            '<div class="section"><h2>7. Impact Summary</h2>'
        ]

        if avoided is not None:
            parts.append(
                f"<p><strong>Total Avoided Emissions:</strong> {avoided:,.1f} tCO2e</p>"
            )

        if env_impacts:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Metric</th><th>Value</th><th>Unit</th></tr>")
            for ei in env_impacts:
                parts.append(
                    f"<tr><td>{_esc(str(ei.get('metric', '')))}</td>"
                    f"<td>{_esc(str(ei.get('value', '')))}</td>"
                    f"<td>{_esc(str(ei.get('unit', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_8_benchmark_performance(self, data: Dict[str, Any]) -> str:
        """Build HTML benchmark performance section."""
        bm = data.get("benchmark_performance", {})
        has_bm = bm.get("has_benchmark", False)
        bm_name = bm.get("benchmark_name", "")

        parts: List[str] = [
            '<div class="section"><h2>8. Benchmark Performance</h2>'
        ]

        if has_bm:
            parts.append(f"<p><strong>Benchmark:</strong> {_esc(bm_name)}</p>")
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Metric</th><th>Fund</th><th>Benchmark</th></tr>")
            fund_ret = bm.get("fund_return_pct")
            bm_ret = bm.get("benchmark_return_pct")
            if fund_ret is not None and bm_ret is not None:
                parts.append(
                    f"<tr><td>Return</td><td>{fund_ret:.2f}%</td>"
                    f"<td>{bm_ret:.2f}%</td></tr>"
                )
            waci_f = bm.get("carbon_intensity_fund")
            waci_b = bm.get("carbon_intensity_benchmark")
            if waci_f is not None and waci_b is not None:
                parts.append(
                    f"<tr><td>WACI</td><td>{waci_f:.1f}</td>"
                    f"<td>{waci_b:.1f}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<p>No EU Climate Benchmark designated.</p>")

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
            ".info-table, .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".info-table td, .data-table td, .data-table th { padding: 8px 12px; "
            "border: 1px solid #ddd; }\n"
            ".data-table th { background: #1a5276; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p><strong>{self.RTS_REF}</strong></p>\n'
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
