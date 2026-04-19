"""
CarbonTrajectoryReportTemplate - Carbon intensity trajectory to 2050.

This module implements the carbon trajectory report template for PACK-011
SFDR Article 9 products. It provides a comprehensive carbon intensity
trajectory analysis including Paris alignment assessment, SBT (Science
Based Targets) coverage, Implied Temperature Rise (ITR), carbon budget
consumption, and net zero progress tracking.

Article 9 products with environmental objectives must demonstrate
a credible decarbonization pathway and track progress against
science-based benchmarks and the Paris Agreement goals.

Example:
    >>> template = CarbonTrajectoryReportTemplate()
    >>> data = CarbonTrajectoryReportData(
    ...     fund_info=TrajectoryFundInfo(fund_name="Climate Impact Fund", ...),
    ...     intensity_trajectory=IntensityTrajectory(...),
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

class TrajectoryFundInfo(BaseModel):
    """Fund information for carbon trajectory report."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    reporting_date: str = Field("", description="Reporting date (YYYY-MM-DD)")
    currency: str = Field("EUR", description="Base currency")
    nav: Optional[float] = Field(None, ge=0.0, description="NAV at reporting date")
    total_holdings: int = Field(0, ge=0, description="Number of holdings")
    management_company: str = Field("", description="Management company")
    benchmark_name: str = Field("", description="Climate benchmark name")


class IntensityTrajectory(BaseModel):
    """Carbon intensity trajectory data points."""

    baseline_year: int = Field(2019, ge=2015, le=2030, description="Baseline year")
    baseline_intensity: Optional[float] = Field(
        None, description="Baseline WACI (tCO2e/M revenue)"
    )
    current_year: int = Field(2025, ge=2020, le=2030, description="Current year")
    current_intensity: Optional[float] = Field(
        None, description="Current WACI (tCO2e/M revenue)"
    )
    target_2025: Optional[float] = Field(None, description="2025 target intensity")
    target_2030: Optional[float] = Field(None, description="2030 target intensity")
    target_2040: Optional[float] = Field(None, description="2040 target intensity")
    target_2050: Optional[float] = Field(None, description="2050 target intensity")
    annual_datapoints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Annual data: {year, actual, target, benchmark, unit}",
    )
    scope_coverage: str = Field(
        "Scope 1+2", description="Emission scopes covered"
    )
    methodology: str = Field(
        "", description="Intensity calculation methodology"
    )
    unit: str = Field("tCO2e/M EUR revenue", description="Intensity unit")


class ParisAlignmentAssessment(BaseModel):
    """Paris Agreement alignment assessment."""

    alignment_status: str = Field(
        "aligned",
        description="aligned, below_2c, well_below_2c, not_aligned",
    )
    temperature_scenario: str = Field(
        "1.5C", description="Target temperature scenario"
    )
    scenario_source: str = Field(
        "IEA Net Zero 2050", description="Climate scenario source"
    )
    required_annual_reduction: float = Field(
        7.0, ge=0.0, le=100.0,
        description="Required annual reduction for alignment (%)",
    )
    actual_annual_reduction: Optional[float] = Field(
        None, description="Actual annual reduction achieved (%)"
    )
    cumulative_reduction: Optional[float] = Field(
        None, description="Cumulative reduction since baseline (%)"
    )
    gap_to_target: Optional[float] = Field(
        None, description="Gap to required trajectory (%)"
    )
    assessment_confidence: str = Field(
        "high", description="Confidence level: high, medium, low"
    )
    methodology_notes: str = Field(
        "", description="Assessment methodology notes"
    )


class SBTCoverage(BaseModel):
    """Science Based Targets coverage analysis."""

    portfolio_sbt_coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage of portfolio with approved SBTs",
    )
    committed_sbt_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage committed but not yet approved",
    )
    no_sbt_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage with no SBT commitment",
    )
    sbti_aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage SBTi aligned (1.5C or well-below 2C)",
    )
    near_term_targets: int = Field(0, ge=0, description="Holdings with near-term targets")
    net_zero_targets: int = Field(0, ge=0, description="Holdings with net-zero targets")
    top_holdings_sbt: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top holdings SBT status: {name, weight_pct, sbt_status, target_year}",
    )
    engagement_pipeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="SBT engagement pipeline: {company, status, target_date}",
    )


class ImpliedTemperatureRise(BaseModel):
    """Implied Temperature Rise (ITR) analysis."""

    portfolio_itr: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Portfolio ITR (degrees C)"
    )
    benchmark_itr: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Benchmark ITR (degrees C)"
    )
    itr_methodology: str = Field(
        "", description="ITR calculation methodology"
    )
    itr_provider: str = Field("", description="ITR data provider")
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="ITR data coverage (%)"
    )
    sector_itr: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="ITR by sector: {sector, itr, weight_pct}",
    )
    trend: str = Field(
        "stable", description="ITR trend: improving, stable, worsening"
    )
    previous_itr: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Previous period ITR"
    )


class CarbonBudget(BaseModel):
    """Carbon budget consumption analysis."""

    total_budget: Optional[float] = Field(
        None, description="Total carbon budget allocated (tCO2e)"
    )
    consumed_to_date: Optional[float] = Field(
        None, description="Budget consumed to date (tCO2e)"
    )
    remaining_budget: Optional[float] = Field(
        None, description="Remaining carbon budget (tCO2e)"
    )
    consumption_rate_annual: Optional[float] = Field(
        None, description="Annual consumption rate (tCO2e/year)"
    )
    years_remaining: Optional[int] = Field(
        None, ge=0, description="Estimated years of budget remaining"
    )
    budget_scenario: str = Field(
        "1.5C 50% probability",
        description="Carbon budget scenario used",
    )
    on_budget: bool = Field(
        True, description="Whether consumption is within budget"
    )
    overshoot_risk: str = Field(
        "low", description="Overshoot risk: low, medium, high"
    )


class NetZeroProgress(BaseModel):
    """Net zero commitment progress tracking."""

    net_zero_target_year: Optional[int] = Field(
        None, ge=2030, le=2070, description="Net zero target year"
    )
    interim_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Interim targets: {year, target_pct_reduction, actual_pct_reduction, status}",
    )
    offset_usage_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Percentage of emissions addressed via offsets",
    )
    residual_emissions_plan: str = Field(
        "", description="Plan for addressing residual emissions"
    )
    nzaoa_membership: bool = Field(
        False, description="Net Zero Asset Owner Alliance membership"
    )
    nzami_membership: bool = Field(
        False, description="Net Zero Asset Managers Initiative membership"
    )
    transition_plan_status: str = Field(
        "", description="Transition plan status"
    )
    key_milestones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key milestones: {milestone, target_date, status}",
    )


class CarbonTrajectoryReportData(BaseModel):
    """Complete input data for carbon trajectory report."""

    fund_info: TrajectoryFundInfo
    intensity_trajectory: IntensityTrajectory = Field(
        default_factory=IntensityTrajectory
    )
    paris_alignment: ParisAlignmentAssessment = Field(
        default_factory=ParisAlignmentAssessment
    )
    sbt_coverage: SBTCoverage = Field(default_factory=SBTCoverage)
    implied_temperature: ImpliedTemperatureRise = Field(
        default_factory=ImpliedTemperatureRise
    )
    carbon_budget: CarbonBudget = Field(default_factory=CarbonBudget)
    net_zero_progress: NetZeroProgress = Field(
        default_factory=NetZeroProgress
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class CarbonTrajectoryReportTemplate:
    """
    Carbon intensity trajectory report template for Article 9 products.

    Generates a comprehensive carbon trajectory analysis covering
    intensity trajectory to 2050, Paris alignment assessment, SBT
    coverage, Implied Temperature Rise (ITR), carbon budget analysis,
    and net zero progress tracking.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = CarbonTrajectoryReportTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Decarbonization" in md or "Trajectory" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "carbon_trajectory_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CarbonTrajectoryReportTemplate.

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
        Render carbon trajectory report in the specified format.

        Args:
            data: Report data dictionary matching CarbonTrajectoryReportData schema.
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
        Render carbon trajectory report as Markdown.

        Args:
            data: Report data dictionary matching CarbonTrajectoryReportData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_trajectory(data))
        sections.append(self._md_section_2_paris_alignment(data))
        sections.append(self._md_section_3_sbt_coverage(data))
        sections.append(self._md_section_4_itr(data))
        sections.append(self._md_section_5_carbon_budget(data))
        sections.append(self._md_section_6_net_zero(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render carbon trajectory report as self-contained HTML.

        Args:
            data: Report data dictionary matching CarbonTrajectoryReportData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_trajectory(data))
        sections.append(self._html_section_2_paris(data))
        sections.append(self._html_section_3_sbt(data))
        sections.append(self._html_section_4_itr(data))
        sections.append(self._html_section_5_budget(data))
        sections.append(self._html_section_6_net_zero(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 Carbon Trajectory Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render carbon trajectory report as structured JSON.

        Args:
            data: Report data dictionary matching CarbonTrajectoryReportData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_carbon_trajectory",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_info": data.get("fund_info", {}),
            "intensity_trajectory": data.get("intensity_trajectory", {}),
            "paris_alignment": data.get("paris_alignment", {}),
            "sbt_coverage": data.get("sbt_coverage", {}),
            "implied_temperature": data.get("implied_temperature", {}),
            "carbon_budget": data.get("carbon_budget", {}),
            "net_zero_progress": data.get("net_zero_progress", {}),
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
        return (
            f"# Carbon Trajectory Report (SFDR Article 9)\n\n"
            f"**Fund:** {name}\n\n"
            f"**Benchmark:** {fi.get('benchmark_name', 'N/A') or 'N/A'}\n\n"
            f"**Reporting Date:** {fi.get('reporting_date', '')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_trajectory(self, data: Dict[str, Any]) -> str:
        """Section 1: Intensity trajectory to 2050."""
        it = data.get("intensity_trajectory", {})
        baseline_year = it.get("baseline_year", 2019)
        baseline = it.get("baseline_intensity")
        current_year = it.get("current_year", 2025)
        current = it.get("current_intensity")
        t2025 = it.get("target_2025")
        t2030 = it.get("target_2030")
        t2040 = it.get("target_2040")
        t2050 = it.get("target_2050")
        datapoints = it.get("annual_datapoints", [])
        scope = it.get("scope_coverage", "Scope 1+2")
        unit = it.get("unit", "tCO2e/M EUR revenue")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        lines: List[str] = [
            "## 1. Carbon Intensity Trajectory\n",
            f"**Scope:** {scope} | **Unit:** {unit}\n",
            "| Milestone | Intensity |",
            "|-----------|----------|",
            f"| Baseline ({baseline_year}) | {_fmt(baseline)} |",
            f"| Current ({current_year}) | {_fmt(current)} |",
            f"| Target 2025 | {_fmt(t2025)} |",
            f"| Target 2030 | {_fmt(t2030)} |",
            f"| Target 2040 | {_fmt(t2040)} |",
            f"| Target 2050 | {_fmt(t2050)} |",
            "",
        ]

        if datapoints:
            lines.append("### Annual Data Points\n")
            lines.append("| Year | Actual | Target | Benchmark | Unit |")
            lines.append("|------|--------|--------|-----------|------|")
            for dp in datapoints:
                actual = dp.get("actual")
                target = dp.get("target")
                bm = dp.get("benchmark")
                lines.append(
                    f"| {dp.get('year', '')} | "
                    f"{_fmt(actual)} | "
                    f"{_fmt(target)} | "
                    f"{_fmt(bm)} | "
                    f"{dp.get('unit', unit)} |"
                )
            lines.append("")

        # ASCII trajectory chart
        lines.append("### Trajectory Visualization\n")
        lines.append("```")
        chart_points = []
        if baseline is not None:
            chart_points.append((str(baseline_year), baseline))
        for dp in datapoints:
            a = dp.get("actual")
            if a is not None:
                chart_points.append((str(dp.get("year", "")), a))
        if not chart_points and current is not None:
            chart_points.append((str(current_year), current))
        if t2030 is not None:
            chart_points.append(("2030(T)", t2030))
        if t2050 is not None:
            chart_points.append(("2050(T)", t2050))

        max_val = max((v for _, v in chart_points), default=1) or 1
        for label, val in chart_points:
            bar_len = int((val / max_val) * 40)
            bar = "#" * bar_len
            lines.append(f"  {label:10s} [{bar:<40s}] {val:,.1f}")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_2_paris_alignment(self, data: Dict[str, Any]) -> str:
        """Section 2: Paris alignment assessment."""
        pa = data.get("paris_alignment", {})
        status = pa.get("alignment_status", "aligned")
        scenario = pa.get("temperature_scenario", "1.5C")
        source = pa.get("scenario_source", "")
        required = pa.get("required_annual_reduction", 7.0)
        actual = pa.get("actual_annual_reduction")
        cumulative = pa.get("cumulative_reduction")
        gap = pa.get("gap_to_target")
        confidence = pa.get("assessment_confidence", "high")
        notes = pa.get("methodology_notes", "")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.2f}%" if v is not None else "N/A"

        status_display = {
            "aligned": "ALIGNED",
            "below_2c": "BELOW 2C",
            "well_below_2c": "WELL BELOW 2C",
            "not_aligned": "NOT ALIGNED",
        }

        lines: List[str] = [
            "## 2. Paris Agreement Alignment\n",
            f"**Status:** {status_display.get(status, status.upper())} | "
            f"**Scenario:** {scenario} | **Confidence:** {confidence.upper()}\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Scenario Source** | {source} |",
            f"| **Required Annual Reduction** | {required:.1f}% |",
            f"| **Actual Annual Reduction** | {_fmt(actual)} |",
            f"| **Cumulative Reduction** | {_fmt(cumulative)} |",
            f"| **Gap to Target** | {_fmt(gap)} |",
            "",
        ]

        # Visual alignment gauge
        actual_val = actual if actual is not None else 0
        lines.append("### Reduction Progress\n")
        lines.append("```")
        lines.append(
            f"  Required  [{('#' * int(min(required, 100) / 2.5)):<40s}] {required:.1f}%"
        )
        lines.append(
            f"  Actual    [{('#' * int(min(actual_val, 100) / 2.5)):<40s}] {actual_val:.1f}%"
        )
        lines.append("```")

        if notes:
            lines.append(f"\n### Methodology Notes\n\n{notes}")

        return "\n".join(lines)

    def _md_section_3_sbt_coverage(self, data: Dict[str, Any]) -> str:
        """Section 3: Science Based Targets coverage."""
        sbt = data.get("sbt_coverage", {})
        approved = sbt.get("portfolio_sbt_coverage_pct", 0.0)
        committed = sbt.get("committed_sbt_pct", 0.0)
        no_sbt = sbt.get("no_sbt_pct", 0.0)
        sbti_aligned = sbt.get("sbti_aligned_pct", 0.0)
        near_term = sbt.get("near_term_targets", 0)
        net_zero = sbt.get("net_zero_targets", 0)
        top_holdings = sbt.get("top_holdings_sbt", [])
        pipeline = sbt.get("engagement_pipeline", [])

        lines: List[str] = [
            "## 3. Science Based Targets Coverage\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **SBT Approved** | {approved:.1f}% |",
            f"| **SBT Committed** | {committed:.1f}% |",
            f"| **No SBT** | {no_sbt:.1f}% |",
            f"| **SBTi 1.5C Aligned** | {sbti_aligned:.1f}% |",
            f"| **Near-term Targets** | {near_term} holdings |",
            f"| **Net-zero Targets** | {net_zero} holdings |",
            "",
        ]

        # ASCII coverage chart
        lines.append("### Coverage Breakdown\n")
        lines.append("```")
        items = [
            ("Approved SBT", approved),
            ("Committed SBT", committed),
            ("No SBT", no_sbt),
        ]
        for label, pct in items:
            bar_len = int(pct / 2.5)
            bar = "#" * bar_len
            lines.append(f"  {label:16s} [{bar:<40s}] {pct:.1f}%")
        lines.append("```\n")

        if top_holdings:
            lines.append("### Top Holdings SBT Status\n")
            lines.append("| Holding | Weight | SBT Status | Target Year |")
            lines.append("|---------|--------|------------|-------------|")
            for h in top_holdings:
                lines.append(
                    f"| {h.get('name', '')} | "
                    f"{h.get('weight_pct', 0.0):.1f}% | "
                    f"{h.get('sbt_status', '')} | "
                    f"{h.get('target_year', 'N/A')} |"
                )
            lines.append("")

        if pipeline:
            lines.append("### Engagement Pipeline\n")
            lines.append("| Company | Status | Target Date |")
            lines.append("|---------|--------|-------------|")
            for p in pipeline:
                lines.append(
                    f"| {p.get('company', '')} | "
                    f"{p.get('status', '')} | "
                    f"{p.get('target_date', '')} |"
                )

        return "\n".join(lines)

    def _md_section_4_itr(self, data: Dict[str, Any]) -> str:
        """Section 4: Implied Temperature Rise."""
        itr = data.get("implied_temperature", {})
        portfolio_itr = itr.get("portfolio_itr")
        benchmark_itr = itr.get("benchmark_itr")
        methodology = itr.get("itr_methodology", "")
        provider = itr.get("itr_provider", "")
        coverage = itr.get("coverage_pct", 0.0)
        sector_itr = itr.get("sector_itr", [])
        trend = itr.get("trend", "stable")
        previous = itr.get("previous_itr")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.2f}C" if v is not None else "N/A"

        trend_display = {"improving": "IMPROVING", "stable": "STABLE", "worsening": "WORSENING"}

        lines: List[str] = [
            "## 4. Implied Temperature Rise (ITR)\n",
            f"**Portfolio ITR:** {_fmt(portfolio_itr)} | "
            f"**Benchmark ITR:** {_fmt(benchmark_itr)} | "
            f"**Trend:** {trend_display.get(trend, trend.upper())}\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Portfolio ITR** | {_fmt(portfolio_itr)} |",
            f"| **Benchmark ITR** | {_fmt(benchmark_itr)} |",
            f"| **Previous ITR** | {_fmt(previous)} |",
            f"| **Data Coverage** | {coverage:.1f}% |",
            f"| **Provider** | {provider or 'N/A'} |",
            "",
        ]

        if sector_itr:
            lines.append("### ITR by Sector\n")
            lines.append("| Sector | ITR | Weight (%) |")
            lines.append("|--------|-----|-----------|")
            for s in sector_itr:
                s_itr = s.get("itr")
                s_str = f"{s_itr:.2f}C" if s_itr is not None else "N/A"
                lines.append(
                    f"| {s.get('sector', '')} | "
                    f"{s_str} | "
                    f"{s.get('weight_pct', 0.0):.1f}% |"
                )
            lines.append("")

        if methodology:
            lines.append(f"### Methodology\n\n{methodology}")

        return "\n".join(lines)

    def _md_section_5_carbon_budget(self, data: Dict[str, Any]) -> str:
        """Section 5: Carbon budget consumption."""
        cb = data.get("carbon_budget", {})
        total = cb.get("total_budget")
        consumed = cb.get("consumed_to_date")
        remaining = cb.get("remaining_budget")
        rate = cb.get("consumption_rate_annual")
        years = cb.get("years_remaining")
        scenario = cb.get("budget_scenario", "")
        on_budget = cb.get("on_budget", True)
        overshoot = cb.get("overshoot_risk", "low")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.0f}" if v is not None else "N/A"

        lines: List[str] = [
            "## 5. Carbon Budget\n",
            f"**Scenario:** {scenario}\n",
            f"**On Budget:** {'Yes' if on_budget else 'No'} | "
            f"**Overshoot Risk:** {overshoot.upper()}\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Budget** | {_fmt(total)} tCO2e |",
            f"| **Consumed to Date** | {_fmt(consumed)} tCO2e |",
            f"| **Remaining** | {_fmt(remaining)} tCO2e |",
            f"| **Annual Rate** | {_fmt(rate)} tCO2e/yr |",
            f"| **Years Remaining** | {years if years is not None else 'N/A'} |",
            "",
        ]

        # Budget consumption bar
        if total is not None and consumed is not None and total > 0:
            consumed_pct = min((consumed / total) * 100, 100)
            lines.append("### Budget Consumption\n")
            lines.append("```")
            bar_len = int(consumed_pct / 2.5)
            bar = "#" * bar_len
            remaining_bar = "." * (40 - bar_len)
            lines.append(f"  Consumed  [{bar}{remaining_bar}] {consumed_pct:.1f}%")
            lines.append("```")

        return "\n".join(lines)

    def _md_section_6_net_zero(self, data: Dict[str, Any]) -> str:
        """Section 6: Net zero progress."""
        nz = data.get("net_zero_progress", {})
        target_year = nz.get("net_zero_target_year")
        interim = nz.get("interim_targets", [])
        offset = nz.get("offset_usage_pct", 0.0)
        residual = nz.get("residual_emissions_plan", "")
        nzaoa = nz.get("nzaoa_membership", False)
        nzami = nz.get("nzami_membership", False)
        transition = nz.get("transition_plan_status", "")
        milestones = nz.get("key_milestones", [])

        lines: List[str] = [
            "## 6. Net Zero Progress\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Net Zero Target Year** | {target_year if target_year else 'Not set'} |",
            f"| **Offset Usage** | {offset:.1f}% |",
            f"| **NZAOA Membership** | {'Yes' if nzaoa else 'No'} |",
            f"| **NZAMI Membership** | {'Yes' if nzami else 'No'} |",
            f"| **Transition Plan** | {transition or 'N/A'} |",
            "",
        ]

        if interim:
            lines.append("### Interim Targets\n")
            lines.append("| Year | Target Reduction | Actual Reduction | Status |")
            lines.append("|------|-----------------|------------------|--------|")
            for it in interim:
                t = it.get("target_pct_reduction")
                a = it.get("actual_pct_reduction")
                t_str = f"{t:.1f}%" if t is not None else "N/A"
                a_str = f"{a:.1f}%" if a is not None else "N/A"
                lines.append(
                    f"| {it.get('year', '')} | "
                    f"{t_str} | "
                    f"{a_str} | "
                    f"{it.get('status', '')} |"
                )
            lines.append("")

        if milestones:
            lines.append("### Key Milestones\n")
            lines.append("| Milestone | Target Date | Status |")
            lines.append("|-----------|-------------|--------|")
            for m in milestones:
                lines.append(
                    f"| {m.get('milestone', '')} | "
                    f"{m.get('target_date', '')} | "
                    f"{m.get('status', '')} |"
                )
            lines.append("")

        if residual:
            lines.append(f"### Residual Emissions Plan\n\n{residual}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_trajectory(self, data: Dict[str, Any]) -> str:
        """Build HTML trajectory section."""
        it = data.get("intensity_trajectory", {})
        datapoints = it.get("annual_datapoints", [])

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        parts: List[str] = [
            '<div class="section"><h2>1. Carbon Intensity Trajectory</h2>',
        ]

        if datapoints:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Year</th><th>Actual</th><th>Target</th><th>Benchmark</th></tr>")
            for dp in datapoints:
                parts.append(
                    f"<tr><td>{dp.get('year', '')}</td>"
                    f"<td>{_fmt(dp.get('actual'))}</td>"
                    f"<td>{_fmt(dp.get('target'))}</td>"
                    f"<td>{_fmt(dp.get('benchmark'))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_paris(self, data: Dict[str, Any]) -> str:
        """Build HTML Paris alignment section."""
        pa = data.get("paris_alignment", {})
        status = pa.get("alignment_status", "aligned")
        colors = {
            "aligned": "#27ae60", "well_below_2c": "#27ae60",
            "below_2c": "#f39c12", "not_aligned": "#e74c3c",
        }
        color = colors.get(status, "#7f8c8d")

        parts: List[str] = [
            '<div class="section"><h2>2. Paris Agreement Alignment</h2>',
            f'<p>Status: <span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:3px;font-weight:bold;">'
            f'{_esc(status.upper().replace("_", " "))}</span></p>',
            "</div>",
        ]
        return "".join(parts)

    def _html_section_3_sbt(self, data: Dict[str, Any]) -> str:
        """Build HTML SBT coverage section."""
        sbt = data.get("sbt_coverage", {})
        approved = sbt.get("portfolio_sbt_coverage_pct", 0.0)
        committed = sbt.get("committed_sbt_pct", 0.0)

        parts: List[str] = [
            '<div class="section"><h2>3. SBT Coverage</h2>',
            f"<p>Approved: <strong>{approved:.1f}%</strong> | "
            f"Committed: <strong>{committed:.1f}%</strong></p>",
            "</div>",
        ]
        return "".join(parts)

    def _html_section_4_itr(self, data: Dict[str, Any]) -> str:
        """Build HTML ITR section."""
        itr = data.get("implied_temperature", {})
        portfolio_itr = itr.get("portfolio_itr")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.2f}C" if v is not None else "N/A"

        color = "#27ae60"
        if portfolio_itr is not None:
            if portfolio_itr > 2.0:
                color = "#e74c3c"
            elif portfolio_itr > 1.5:
                color = "#f39c12"

        parts: List[str] = [
            '<div class="section"><h2>4. Implied Temperature Rise</h2>',
            f'<p style="font-size:2em;font-weight:bold;color:{color};text-align:center;">'
            f'{_fmt(portfolio_itr)}</p>',
            "</div>",
        ]
        return "".join(parts)

    def _html_section_5_budget(self, data: Dict[str, Any]) -> str:
        """Build HTML carbon budget section."""
        cb = data.get("carbon_budget", {})
        on_budget = cb.get("on_budget", True)
        overshoot = cb.get("overshoot_risk", "low")

        colors = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
        color = colors.get(overshoot, "#7f8c8d")

        parts: List[str] = [
            '<div class="section"><h2>5. Carbon Budget</h2>',
            f'<p>On Budget: <strong>{"Yes" if on_budget else "No"}</strong> | '
            f'Overshoot Risk: <span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:3px;">{_esc(overshoot.upper())}</span></p>',
            "</div>",
        ]
        return "".join(parts)

    def _html_section_6_net_zero(self, data: Dict[str, Any]) -> str:
        """Build HTML net zero section."""
        nz = data.get("net_zero_progress", {})
        target_year = nz.get("net_zero_target_year")
        milestones = nz.get("key_milestones", [])

        parts: List[str] = [
            '<div class="section"><h2>6. Net Zero Progress</h2>',
            f"<p>Target Year: <strong>{target_year if target_year else 'Not set'}</strong></p>",
        ]

        if milestones:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Milestone</th><th>Target Date</th><th>Status</th></tr>")
            for m in milestones:
                parts.append(
                    f"<tr><td>{_esc(str(m.get('milestone', '')))}</td>"
                    f"<td>{_esc(str(m.get('target_date', '')))}</td>"
                    f"<td>{_esc(str(m.get('status', '')))}</td></tr>"
                )
            parts.append("</table>")

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
