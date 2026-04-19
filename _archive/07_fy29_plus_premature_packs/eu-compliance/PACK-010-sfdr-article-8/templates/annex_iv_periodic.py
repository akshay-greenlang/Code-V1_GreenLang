"""
AnnexIVPeriodicTemplate - SFDR RTS Annex IV periodic reporting template.

This module implements the Annex IV periodic reporting template for
PACK-010 SFDR Article 8 products. It generates the mandatory periodic
disclosure required under SFDR Delegated Regulation (EU) 2022/1288,
covering 8 sections from product identification through benchmark
comparison.

The template compares actual E/S characteristic attainment against
targets and provides year-over-year PAI indicator tracking.

Example:
    >>> template = AnnexIVPeriodicTemplate()
    >>> data = PeriodicData(
    ...     reporting_period=ReportingPeriod(start_date="2025-01-01", ...),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

from __future__ import annotations

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

class ReportingPeriod(BaseModel):
    """Reporting period identification."""

    start_date: str = Field(..., description="Period start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Period end date (YYYY-MM-DD)")
    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    lei: str = Field("", description="Legal Entity Identifier")
    management_company: str = Field("", description="Management company name")
    sfdr_classification: str = Field("article_8", description="SFDR classification")
    fund_currency: str = Field("EUR", description="Fund base currency")
    nav_end_period: Optional[float] = Field(None, ge=0.0, description="NAV at period end")

    @field_validator("sfdr_classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        """Validate classification."""
        allowed = {"article_8", "article_8_plus", "article_9"}
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v


class CharacteristicAttainment(BaseModel):
    """Attainment of an E/S characteristic during the reporting period."""

    characteristic_name: str = Field("", description="Characteristic name")
    characteristic_type: str = Field(
        "environmental", description="environmental or social"
    )
    target: Optional[float] = Field(None, description="Target value (if quantitative)")
    target_description: str = Field("", description="Target description")
    actual: Optional[float] = Field(None, description="Actual achieved value")
    actual_description: str = Field("", description="Actual outcome description")
    attained_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Attainment percentage"
    )
    indicator_used: str = Field("", description="Sustainability indicator used")
    commentary: str = Field("", description="Commentary on attainment")


class TopInvestment(BaseModel):
    """Top investment holding."""

    rank: int = Field(0, ge=1, description="Rank position")
    name: str = Field("", description="Investment name")
    sector: str = Field("", description="Sector / NACE code")
    country: str = Field("", description="Country of domicile")
    weight_pct: float = Field(0.0, ge=0.0, le=100.0, description="Portfolio weight %")
    esg_score: Optional[float] = Field(None, description="ESG score")
    sustainable: bool = Field(False, description="Qualifies as sustainable investment")
    taxonomy_aligned: bool = Field(False, description="Taxonomy-aligned")


class ProportionBreakdown(BaseModel):
    """Proportion breakdown for the reporting period."""

    sustainable_total: float = Field(
        0.0, ge=0.0, le=100.0, description="Total sustainable %"
    )
    taxonomy_aligned: float = Field(
        0.0, ge=0.0, le=100.0, description="Taxonomy-aligned %"
    )
    other_env: float = Field(
        0.0, ge=0.0, le=100.0, description="Other environmental %"
    )
    social: float = Field(0.0, ge=0.0, le=100.0, description="Social %")
    not_sustainable: float = Field(
        0.0, ge=0.0, le=100.0, description="Not sustainable %"
    )
    previous_sustainable_total: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Previous period sustainable %"
    )
    previous_taxonomy_aligned: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Previous period taxonomy %"
    )


class PAISummary(BaseModel):
    """Principal Adverse Impact indicator summary."""

    indicator_id: int = Field(0, ge=1, le=18, description="PAI indicator number (1-18)")
    indicator_name: str = Field("", description="Indicator name per RTS")
    metric: str = Field("", description="Metric description")
    value: float = Field(0.0, description="Current period value")
    unit: str = Field("", description="Unit of measurement")
    previous_value: Optional[float] = Field(None, description="Previous period value")
    yoy_change: Optional[float] = Field(None, description="Year-over-year change")
    explanation: str = Field("", description="Explanation of value")
    actions_taken: str = Field("", description="Actions taken to address")


class ActionTaken(BaseModel):
    """Action taken to meet E/S characteristics during the period."""

    action_description: str = Field("", description="Description of action")
    category: str = Field("", description="Category: engagement, exclusion, voting, etc.")
    outcome: str = Field("", description="Outcome of the action")
    quantitative_impact: Optional[str] = Field(None, description="Quantitative impact")


class BenchmarkComparison(BaseModel):
    """Comparison against designated benchmark."""

    benchmark_name: str = Field("", description="Benchmark name")
    fund_return_pct: Optional[float] = Field(None, description="Fund return %")
    benchmark_return_pct: Optional[float] = Field(None, description="Benchmark return %")
    esg_score_fund: Optional[float] = Field(None, description="Fund ESG score")
    esg_score_benchmark: Optional[float] = Field(None, description="Benchmark ESG score")
    tracking_error: Optional[float] = Field(None, description="Tracking error")
    commentary: str = Field("", description="Benchmark comparison commentary")


class PeriodicData(BaseModel):
    """Complete input data for Annex IV periodic report."""

    reporting_period: ReportingPeriod
    characteristic_attainments: List[CharacteristicAttainment] = Field(
        default_factory=list
    )
    top_investments: List[TopInvestment] = Field(default_factory=list)
    proportion_breakdown: ProportionBreakdown = Field(
        default_factory=ProportionBreakdown
    )
    pai_summaries: List[PAISummary] = Field(default_factory=list)
    actions_taken: List[ActionTaken] = Field(default_factory=list)
    benchmark_comparison: Optional[BenchmarkComparison] = Field(None)
    narrative_summary: str = Field("", description="Overall narrative summary")


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AnnexIVPeriodicTemplate:
    """
    SFDR RTS Annex IV periodic reporting template for Article 8 products.

    Generates the mandatory periodic disclosure required under SFDR
    Delegated Regulation (EU) 2022/1288 Annex IV. Reports on how E/S
    characteristics were attained during the reference period.

    Attributes:
        config: Optional configuration dictionary.

    Example:
        >>> template = AnnexIVPeriodicTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Periodic Report" in md
    """

    PACK_ID = "PACK-010"
    TEMPLATE_NAME = "annex_iv_periodic"
    VERSION = "1.0"
    REGULATION_REF = "Regulation (EU) 2019/2088"
    RTS_REF = "Delegated Regulation (EU) 2022/1288, Annex IV"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnnexIVPeriodicTemplate."""
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    #  Public render dispatcher
    # ------------------------------------------------------------------ #

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render periodic report in the specified format.

        Args:
            data: Report data dictionary matching PeriodicData schema.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered content.

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
        Render the Annex IV periodic report as Markdown.

        Args:
            data: Report data dictionary matching PeriodicData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = [
            self._md_header(data),
            self._md_section_1_product_id(data),
            self._md_section_2_characteristic_attainment(data),
            self._md_section_3_top_investments(data),
            self._md_section_4_proportions(data),
            self._md_section_5_pai_indicators(data),
            self._md_section_6_actions_taken(data),
            self._md_section_7_comparison(data),
            self._md_section_8_benchmark(data),
        ]

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(provenance_hash)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the Annex IV periodic report as self-contained HTML.

        Args:
            data: Report data dictionary matching PeriodicData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = [
            self._html_section_1_product_id(data),
            self._html_section_2_attainment(data),
            self._html_section_3_top_investments(data),
            self._html_section_4_proportions(data),
            self._html_section_5_pai(data),
            self._html_section_6_actions(data),
            self._html_section_7_comparison(data),
            self._html_section_8_benchmark(data),
        ]

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Annex IV Periodic Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the Annex IV periodic report as structured JSON.

        Args:
            data: Report data dictionary matching PeriodicData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        rp = data.get("reporting_period", {})
        report: Dict[str, Any] = {
            "report_type": "sfdr_annex_iv_periodic",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "regulation": self.REGULATION_REF,
            "rts_annex": self.RTS_REF,
            "reporting_period": {
                "start_date": rp.get("start_date", ""),
                "end_date": rp.get("end_date", ""),
                "fund_name": rp.get("fund_name", ""),
                "isin": rp.get("isin", ""),
                "lei": rp.get("lei", ""),
                "nav_end_period": rp.get("nav_end_period"),
            },
            "characteristic_attainments": data.get("characteristic_attainments", []),
            "top_investments": data.get("top_investments", []),
            "proportion_breakdown": data.get("proportion_breakdown", {}),
            "pai_summaries": data.get("pai_summaries", []),
            "actions_taken": data.get("actions_taken", []),
            "benchmark_comparison": data.get("benchmark_comparison"),
            "narrative_summary": data.get("narrative_summary", ""),
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
        return (
            f"# SFDR Periodic Report (Article 8)\n\n"
            f"**{self.RTS_REF}**\n\n"
            f"**Fund:** {rp.get('fund_name', 'Unknown')}\n\n"
            f"**Period:** {rp.get('start_date', '')} to {rp.get('end_date', '')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_product_id(self, data: Dict[str, Any]) -> str:
        """Section 1: Product identification."""
        rp = data.get("reporting_period", {})
        nav = rp.get("nav_end_period")
        nav_str = f"{nav:,.2f} {rp.get('fund_currency', 'EUR')}" if nav else "N/A"

        lines = [
            "## 1. Product Identification\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Fund Name** | {rp.get('fund_name', 'N/A')} |",
            f"| **ISIN** | {rp.get('isin', '') or 'N/A'} |",
            f"| **LEI** | {rp.get('lei', '') or 'N/A'} |",
            f"| **Classification** | {self._format_classification(rp.get('sfdr_classification', ''))} |",
            f"| **Management Company** | {rp.get('management_company', '') or 'N/A'} |",
            f"| **Reporting Period** | {rp.get('start_date', '')} to {rp.get('end_date', '')} |",
            f"| **NAV (end of period)** | {nav_str} |",
        ]
        return "\n".join(lines)

    def _md_section_2_characteristic_attainment(self, data: Dict[str, Any]) -> str:
        """Section 2: E/S characteristics attainment."""
        attainments = data.get("characteristic_attainments", [])

        lines = [
            "## 2. E/S Characteristics Attainment\n",
            "How did the financial product perform with regard to the environmental "
            "and/or social characteristics it promotes?\n",
        ]

        if attainments:
            lines.append(
                "| Characteristic | Type | Target | Actual | Attainment | Indicator |"
            )
            lines.append(
                "|----------------|------|--------|--------|------------|-----------|"
            )
            for a in attainments:
                target = (
                    f"{a.get('target', 0):.2f}"
                    if a.get("target") is not None
                    else a.get("target_description", "N/A")
                )
                actual = (
                    f"{a.get('actual', 0):.2f}"
                    if a.get("actual") is not None
                    else a.get("actual_description", "N/A")
                )
                attained = (
                    f"{a.get('attained_pct', 0):.1f}%"
                    if a.get("attained_pct") is not None
                    else "N/A"
                )
                att_icon = self._attainment_icon(a.get("attained_pct"))
                lines.append(
                    f"| {a.get('characteristic_name', '')} | "
                    f"{a.get('characteristic_type', '')} | "
                    f"{target} | {actual} | {attained} {att_icon} | "
                    f"{a.get('indicator_used', '')} |"
                )
            lines.append("")

            # Detailed commentary
            for a in attainments:
                commentary = a.get("commentary", "")
                if commentary:
                    lines.append(
                        f"**{a.get('characteristic_name', '')}:** {commentary}\n"
                    )
        else:
            lines.append("*No characteristic attainment data available.*")

        return "\n".join(lines)

    def _md_section_3_top_investments(self, data: Dict[str, Any]) -> str:
        """Section 3: Top 15 investments."""
        investments = data.get("top_investments", [])[:15]

        lines = [
            "## 3. Top Investments\n",
            "The 15 largest investments during the reference period:\n",
            "| # | Investment | Sector | Country | Weight | ESG Score | Sustainable | Taxonomy |",
            "|---|-----------|--------|---------|--------|-----------|-------------|----------|",
        ]

        for inv in investments:
            esg = f"{inv.get('esg_score', 0):.1f}" if inv.get("esg_score") is not None else "N/A"
            sust = "Yes" if inv.get("sustainable") else "No"
            tax = "Yes" if inv.get("taxonomy_aligned") else "No"
            lines.append(
                f"| {inv.get('rank', 0)} | "
                f"{inv.get('name', '')} | "
                f"{inv.get('sector', '')} | "
                f"{inv.get('country', '')} | "
                f"{inv.get('weight_pct', 0.0):.2f}% | "
                f"{esg} | {sust} | {tax} |"
            )

        if not investments:
            lines.append("| *No investment data available* | | | | | | | |")

        return "\n".join(lines)

    def _md_section_4_proportions(self, data: Dict[str, Any]) -> str:
        """Section 4: Proportion of sustainability-related investments."""
        pb = data.get("proportion_breakdown", {})
        prev_sust = pb.get("previous_sustainable_total")
        prev_tax = pb.get("previous_taxonomy_aligned")

        lines = [
            "## 4. Proportion of Sustainability-Related Investments\n",
            "| Category | Current Period | Previous Period | Change |",
            "|----------|---------------|-----------------|--------|",
        ]

        current_sust = pb.get("sustainable_total", 0.0)
        current_tax = pb.get("taxonomy_aligned", 0.0)

        sust_change = self._format_change(current_sust, prev_sust)
        tax_change = self._format_change(current_tax, prev_tax)

        lines.append(
            f"| **Sustainable Total** | {current_sust:.1f}% | "
            f"{prev_sust:.1f}% | {sust_change} |"
            if prev_sust is not None
            else f"| **Sustainable Total** | {current_sust:.1f}% | N/A | N/A |"
        )
        lines.append(
            f"| Taxonomy-aligned | {current_tax:.1f}% | "
            f"{prev_tax:.1f}% | {tax_change} |"
            if prev_tax is not None
            else f"| Taxonomy-aligned | {current_tax:.1f}% | N/A | N/A |"
        )
        lines.append(
            f"| Other environmental | {pb.get('other_env', 0.0):.1f}% | - | - |"
        )
        lines.append(f"| Social | {pb.get('social', 0.0):.1f}% | - | - |")
        lines.append(
            f"| Not sustainable | {pb.get('not_sustainable', 0.0):.1f}% | - | - |"
        )

        # Bar chart
        lines.append("\n### Allocation Comparison\n")
        lines.append("```")
        items = [
            ("Taxonomy-aligned", current_tax),
            ("Other env.", pb.get("other_env", 0.0)),
            ("Social", pb.get("social", 0.0)),
            ("Not sustainable", pb.get("not_sustainable", 0.0)),
        ]
        for label, pct in items:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<50s}] {pct:.1f}%")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_5_pai_indicators(self, data: Dict[str, Any]) -> str:
        """Section 5: PAI indicators considered."""
        pais = data.get("pai_summaries", [])

        lines = [
            "## 5. Principal Adverse Impact (PAI) Indicators\n",
            "| # | Indicator | Value | Unit | Previous | YoY Change |",
            "|---|-----------|-------|------|----------|------------|",
        ]

        for p in pais:
            prev = (
                f"{p.get('previous_value', 0):.4f}"
                if p.get("previous_value") is not None
                else "N/A"
            )
            yoy = (
                f"{p.get('yoy_change', 0):+.2f}%"
                if p.get("yoy_change") is not None
                else "N/A"
            )
            lines.append(
                f"| {p.get('indicator_id', 0)} | "
                f"{p.get('indicator_name', '')} | "
                f"{p.get('value', 0):.4f} | "
                f"{p.get('unit', '')} | "
                f"{prev} | {yoy} |"
            )

        if not pais:
            lines.append("| *No PAI data available* | | | | | |")

        # Detailed explanations
        lines.append("")
        for p in pais:
            explanation = p.get("explanation", "")
            actions = p.get("actions_taken", "")
            if explanation or actions:
                lines.append(f"### PAI #{p.get('indicator_id', 0)}: {p.get('indicator_name', '')}\n")
                if explanation:
                    lines.append(f"**Explanation:** {explanation}\n")
                if actions:
                    lines.append(f"**Actions Taken:** {actions}\n")

        return "\n".join(lines)

    def _md_section_6_actions_taken(self, data: Dict[str, Any]) -> str:
        """Section 6: Actions taken to meet E/S characteristics."""
        actions = data.get("actions_taken", [])

        lines = [
            "## 6. Actions Taken\n",
            "Actions taken during the reference period to meet the environmental "
            "and/or social characteristics:\n",
        ]

        if actions:
            lines.append("| # | Action | Category | Outcome | Impact |")
            lines.append("|---|--------|----------|---------|--------|")
            for i, a in enumerate(actions, 1):
                impact = a.get("quantitative_impact", "N/A") or "N/A"
                lines.append(
                    f"| {i} | {a.get('action_description', '')} | "
                    f"{a.get('category', '')} | "
                    f"{a.get('outcome', '')} | {impact} |"
                )
        else:
            lines.append("*No specific actions recorded for this period.*")

        return "\n".join(lines)

    def _md_section_7_comparison(self, data: Dict[str, Any]) -> str:
        """Section 7: Comparison to previous period."""
        pb = data.get("proportion_breakdown", {})
        narrative = data.get("narrative_summary", "")

        lines = [
            "## 7. Comparison to Previous Period\n",
        ]

        prev_sust = pb.get("previous_sustainable_total")
        current_sust = pb.get("sustainable_total", 0.0)

        if prev_sust is not None:
            diff = current_sust - prev_sust
            direction = "increased" if diff > 0 else "decreased" if diff < 0 else "unchanged"
            lines.append(
                f"The proportion of sustainable investments has **{direction}** "
                f"from {prev_sust:.1f}% to {current_sust:.1f}% "
                f"({diff:+.1f} percentage points).\n"
            )
        else:
            lines.append(
                "No previous period data is available for comparison. "
                "This is the first reporting period.\n"
            )

        if narrative:
            lines.append(f"### Summary\n\n{narrative}")

        return "\n".join(lines)

    def _md_section_8_benchmark(self, data: Dict[str, Any]) -> str:
        """Section 8: Benchmark comparison (if designated)."""
        bc = data.get("benchmark_comparison")

        lines = ["## 8. Benchmark Comparison\n"]

        if bc is None:
            lines.append("No reference benchmark has been designated for this product.")
            return "\n".join(lines)

        lines.append(f"**Benchmark:** {bc.get('benchmark_name', 'N/A')}\n")

        lines.append("| Metric | Fund | Benchmark | Difference |")
        lines.append("|--------|------|-----------|------------|")

        fund_ret = bc.get("fund_return_pct")
        bench_ret = bc.get("benchmark_return_pct")
        if fund_ret is not None and bench_ret is not None:
            diff = fund_ret - bench_ret
            lines.append(
                f"| Return | {fund_ret:.2f}% | {bench_ret:.2f}% | {diff:+.2f}% |"
            )

        fund_esg = bc.get("esg_score_fund")
        bench_esg = bc.get("esg_score_benchmark")
        if fund_esg is not None and bench_esg is not None:
            diff = fund_esg - bench_esg
            lines.append(
                f"| ESG Score | {fund_esg:.1f} | {bench_esg:.1f} | {diff:+.1f} |"
            )

        tracking = bc.get("tracking_error")
        if tracking is not None:
            lines.append(f"| Tracking Error | {tracking:.2f}% | - | - |")

        commentary = bc.get("commentary", "")
        if commentary:
            lines.append(f"\n{commentary}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_product_id(self, data: Dict[str, Any]) -> str:
        """Build HTML product identification section."""
        rp = data.get("reporting_period", {})
        nav = rp.get("nav_end_period")
        nav_str = f"{nav:,.2f} {rp.get('fund_currency', 'EUR')}" if nav else "N/A"

        return (
            '<div class="section">'
            "<h2>1. Product Identification</h2>"
            '<table class="info-table">'
            f"<tr><td><strong>Fund Name</strong></td><td>{_esc(rp.get('fund_name', ''))}</td></tr>"
            f"<tr><td><strong>ISIN</strong></td><td>{_esc(rp.get('isin', '') or 'N/A')}</td></tr>"
            f"<tr><td><strong>Period</strong></td>"
            f"<td>{_esc(rp.get('start_date', ''))} to {_esc(rp.get('end_date', ''))}</td></tr>"
            f"<tr><td><strong>NAV</strong></td><td>{_esc(nav_str)}</td></tr>"
            "</table></div>"
        )

    def _html_section_2_attainment(self, data: Dict[str, Any]) -> str:
        """Build HTML characteristic attainment section."""
        attainments = data.get("characteristic_attainments", [])
        parts = ['<div class="section"><h2>2. E/S Characteristics Attainment</h2>']

        if attainments:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Characteristic</th><th>Type</th><th>Target</th>"
                "<th>Actual</th><th>Attainment</th></tr>"
            )
            for a in attainments:
                target = (
                    f"{a.get('target', 0):.2f}"
                    if a.get("target") is not None
                    else _esc(a.get("target_description", "N/A"))
                )
                actual = (
                    f"{a.get('actual', 0):.2f}"
                    if a.get("actual") is not None
                    else _esc(a.get("actual_description", "N/A"))
                )
                pct = a.get("attained_pct")
                color = self._attainment_color(pct)
                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                parts.append(
                    f"<tr><td>{_esc(a.get('characteristic_name', ''))}</td>"
                    f"<td>{_esc(a.get('characteristic_type', ''))}</td>"
                    f"<td>{target}</td><td>{actual}</td>"
                    f'<td style="color:{color};font-weight:bold;">{pct_str}</td></tr>'
                )
            parts.append("</table>")
        else:
            parts.append("<p><em>No attainment data available.</em></p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_top_investments(self, data: Dict[str, Any]) -> str:
        """Build HTML top investments section."""
        investments = data.get("top_investments", [])[:15]
        parts = ['<div class="section"><h2>3. Top Investments</h2>']

        if investments:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>#</th><th>Investment</th><th>Sector</th>"
                "<th>Country</th><th>Weight</th><th>ESG</th>"
                "<th>Sustainable</th></tr>"
            )
            for inv in investments:
                esg = (
                    f"{inv.get('esg_score', 0):.1f}"
                    if inv.get("esg_score") is not None
                    else "N/A"
                )
                sust = "Yes" if inv.get("sustainable") else "No"
                parts.append(
                    f"<tr><td>{inv.get('rank', 0)}</td>"
                    f"<td>{_esc(inv.get('name', ''))}</td>"
                    f"<td>{_esc(inv.get('sector', ''))}</td>"
                    f"<td>{_esc(inv.get('country', ''))}</td>"
                    f"<td>{inv.get('weight_pct', 0):.2f}%</td>"
                    f"<td>{esg}</td><td>{sust}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_proportions(self, data: Dict[str, Any]) -> str:
        """Build HTML proportions section with visual bar chart."""
        pb = data.get("proportion_breakdown", {})
        parts = ['<div class="section"><h2>4. Proportion of Investments</h2>']

        items = [
            ("Taxonomy-aligned", pb.get("taxonomy_aligned", 0.0), "#2ecc71"),
            ("Other environmental", pb.get("other_env", 0.0), "#27ae60"),
            ("Social", pb.get("social", 0.0), "#3498db"),
            ("Not sustainable", pb.get("not_sustainable", 0.0), "#95a5a6"),
        ]

        for label, pct, color in items:
            bar_width = max(int(pct * 3), 0)
            parts.append(
                f'<div style="margin:8px 0;"><span style="display:inline-block;'
                f'width:160px;">{_esc(label)}</span>'
                f'<div style="display:inline-block;background:{color};'
                f'width:{bar_width}px;height:18px;border-radius:3px;'
                f'vertical-align:middle;margin-right:8px;"></div>'
                f"<span>{pct:.1f}%</span></div>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_pai(self, data: Dict[str, Any]) -> str:
        """Build HTML PAI indicators section."""
        pais = data.get("pai_summaries", [])
        parts = ['<div class="section"><h2>5. PAI Indicators</h2>']

        if pais:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>#</th><th>Indicator</th><th>Value</th>"
                "<th>Unit</th><th>Previous</th><th>YoY</th></tr>"
            )
            for p in pais:
                prev = (
                    f"{p.get('previous_value', 0):.4f}"
                    if p.get("previous_value") is not None
                    else "N/A"
                )
                yoy = p.get("yoy_change")
                yoy_str = f"{yoy:+.2f}%" if yoy is not None else "N/A"
                yoy_color = (
                    "#e74c3c" if yoy and yoy > 0
                    else "#2ecc71" if yoy and yoy < 0
                    else "#2c3e50"
                )
                parts.append(
                    f"<tr><td>{p.get('indicator_id', 0)}</td>"
                    f"<td>{_esc(p.get('indicator_name', ''))}</td>"
                    f"<td>{p.get('value', 0):.4f}</td>"
                    f"<td>{_esc(p.get('unit', ''))}</td>"
                    f"<td>{prev}</td>"
                    f'<td style="color:{yoy_color};font-weight:bold;">{yoy_str}</td></tr>'
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_actions(self, data: Dict[str, Any]) -> str:
        """Build HTML actions taken section."""
        actions = data.get("actions_taken", [])
        parts = ['<div class="section"><h2>6. Actions Taken</h2>']

        if actions:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>#</th><th>Action</th><th>Category</th>"
                "<th>Outcome</th><th>Impact</th></tr>"
            )
            for i, a in enumerate(actions, 1):
                impact = _esc(a.get("quantitative_impact", "") or "N/A")
                parts.append(
                    f"<tr><td>{i}</td>"
                    f"<td>{_esc(a.get('action_description', ''))}</td>"
                    f"<td>{_esc(a.get('category', ''))}</td>"
                    f"<td>{_esc(a.get('outcome', ''))}</td>"
                    f"<td>{impact}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<p><em>No actions recorded.</em></p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_7_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML comparison to previous period."""
        pb = data.get("proportion_breakdown", {})
        narrative = data.get("narrative_summary", "")
        parts = ['<div class="section"><h2>7. Comparison to Previous Period</h2>']

        prev_sust = pb.get("previous_sustainable_total")
        current_sust = pb.get("sustainable_total", 0.0)

        if prev_sust is not None:
            diff = current_sust - prev_sust
            color = "#2ecc71" if diff > 0 else "#e74c3c" if diff < 0 else "#2c3e50"
            parts.append(
                f'<p>Sustainable investments: <strong style="color:{color};">'
                f"{current_sust:.1f}%</strong> (from {prev_sust:.1f}%, "
                f"change: {diff:+.1f}pp)</p>"
            )
        else:
            parts.append("<p>First reporting period. No comparison available.</p>")

        if narrative:
            parts.append(f"<p>{_esc(narrative)}</p>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_8_benchmark(self, data: Dict[str, Any]) -> str:
        """Build HTML benchmark comparison section."""
        bc = data.get("benchmark_comparison")
        parts = ['<div class="section"><h2>8. Benchmark Comparison</h2>']

        if bc is None:
            parts.append("<p>No reference benchmark designated.</p>")
        else:
            parts.append(f"<p><strong>Benchmark:</strong> {_esc(bc.get('benchmark_name', ''))}</p>")
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Metric</th><th>Fund</th><th>Benchmark</th><th>Diff</th></tr>")

            fund_ret = bc.get("fund_return_pct")
            bench_ret = bc.get("benchmark_return_pct")
            if fund_ret is not None and bench_ret is not None:
                diff = fund_ret - bench_ret
                parts.append(
                    f"<tr><td>Return</td><td>{fund_ret:.2f}%</td>"
                    f"<td>{bench_ret:.2f}%</td><td>{diff:+.2f}%</td></tr>"
                )

            fund_esg = bc.get("esg_score_fund")
            bench_esg = bc.get("esg_score_benchmark")
            if fund_esg is not None and bench_esg is not None:
                diff = fund_esg - bench_esg
                parts.append(
                    f"<tr><td>ESG Score</td><td>{fund_esg:.1f}</td>"
                    f"<td>{bench_esg:.1f}</td><td>{diff:+.1f}</td></tr>"
                )

            parts.append("</table>")

            commentary = bc.get("commentary", "")
            if commentary:
                parts.append(f"<p>{_esc(commentary)}</p>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _attainment_icon(pct: Optional[float]) -> str:
        """Return a text icon based on attainment percentage."""
        if pct is None:
            return ""
        if pct >= 100:
            return "[FULL]"
        if pct >= 80:
            return "[HIGH]"
        if pct >= 50:
            return "[MED]"
        return "[LOW]"

    @staticmethod
    def _attainment_color(pct: Optional[float]) -> str:
        """Return color based on attainment percentage."""
        if pct is None:
            return "#2c3e50"
        if pct >= 100:
            return "#2ecc71"
        if pct >= 80:
            return "#27ae60"
        if pct >= 50:
            return "#f39c12"
        return "#e74c3c"

    @staticmethod
    def _format_change(current: float, previous: Optional[float]) -> str:
        """Format change between periods."""
        if previous is None:
            return "N/A"
        diff = current - previous
        return f"{diff:+.1f}pp"

    @staticmethod
    def _format_classification(classification: str) -> str:
        """Format SFDR classification for display."""
        mapping = {
            "article_8": "Article 8 (Light Green)",
            "article_8_plus": "Article 8+ (with Sustainable Investment)",
            "article_9": "Article 9 (Dark Green)",
        }
        return mapping.get(classification, classification)

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
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px auto; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; }\n"
            ".section { margin-bottom: 25px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".info-table, .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".info-table td, .data-table td, .data-table th { padding: 8px 12px; "
            "border: 1px solid #ddd; }\n"
            ".data-table th { background: #2c3e50; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 20px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f"<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID}</div>\n'
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
