"""
BenchmarkMethodologyTemplate - CTB/PAB benchmark methodology report.

This module implements the benchmark methodology template for PACK-011
SFDR Article 9 products. It generates the EU Climate Benchmark analysis
covering CTB/PAB selection rationale, carbon intensity comparison,
decarbonization trajectory, exclusion compliance, and tracking error.

Article 9 products that designate a Paris-Aligned Benchmark (PAB) or
Climate Transition Benchmark (CTB) must disclose how the benchmark
methodology aligns with their sustainable investment objective.

Example:
    >>> template = BenchmarkMethodologyTemplate()
    >>> data = BenchmarkMethodologyData(
    ...     fund_info=BenchmarkFundInfo(fund_name="Climate Impact Fund", ...),
    ...     benchmark_selection=BenchmarkSelection(...),
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

class BenchmarkFundInfo(BaseModel):
    """Fund information for benchmark methodology report."""

    fund_name: str = Field(..., min_length=1, description="Fund name")
    isin: str = Field("", description="ISIN code")
    benchmark_name: str = Field("", description="Designated benchmark name")
    benchmark_provider: str = Field("", description="Benchmark index provider")
    reporting_date: str = Field("", description="Reporting date (YYYY-MM-DD)")
    currency: str = Field("EUR", description="Base currency")
    nav: Optional[float] = Field(None, ge=0.0, description="NAV at reporting date")
    management_company: str = Field("", description="Management company")


class BenchmarkSelection(BaseModel):
    """CTB/PAB selection rationale."""

    benchmark_type: str = Field(
        "PAB",
        description="Benchmark type: CTB or PAB",
    )
    selection_rationale: str = Field(
        "", description="Rationale for selecting this benchmark type"
    )
    alignment_with_objective: str = Field(
        "",
        description="How the benchmark aligns with the sustainable investment objective",
    )
    eu_regulation_reference: str = Field(
        "Regulation (EU) 2020/1818",
        description="EU benchmark regulation reference",
    )
    ctb_compliance_criteria: List[str] = Field(
        default_factory=list,
        description="CTB-specific compliance criteria met",
    )
    pab_compliance_criteria: List[str] = Field(
        default_factory=list,
        description="PAB-specific compliance criteria met",
    )
    deviation_from_parent: str = Field(
        "",
        description="Key deviations from the parent index",
    )
    rebalancing_frequency: str = Field(
        "quarterly", description="Benchmark rebalancing frequency"
    )

    @field_validator("benchmark_type")
    @classmethod
    def validate_benchmark_type(cls, v: str) -> str:
        """Validate benchmark type is CTB or PAB."""
        allowed = {"CTB", "PAB"}
        if v not in allowed:
            raise ValueError(f"benchmark_type must be one of {allowed}, got '{v}'")
        return v


class CarbonIntensityComparison(BaseModel):
    """Carbon intensity comparison between fund and benchmark."""

    fund_waci: Optional[float] = Field(
        None, description="Fund WACI (tCO2e/M revenue)"
    )
    benchmark_waci: Optional[float] = Field(
        None, description="Benchmark WACI (tCO2e/M revenue)"
    )
    parent_index_waci: Optional[float] = Field(
        None, description="Parent index WACI (tCO2e/M revenue)"
    )
    reduction_vs_parent: Optional[float] = Field(
        None, description="Percentage reduction vs parent index"
    )
    fund_carbon_footprint: Optional[float] = Field(
        None, description="Fund carbon footprint (tCO2e/M invested)"
    )
    benchmark_carbon_footprint: Optional[float] = Field(
        None, description="Benchmark carbon footprint (tCO2e/M invested)"
    )
    fund_total_emissions: Optional[float] = Field(
        None, description="Fund total financed emissions (tCO2e)"
    )
    benchmark_total_emissions: Optional[float] = Field(
        None, description="Benchmark total financed emissions (tCO2e)"
    )
    scope_coverage: str = Field(
        "Scope 1+2", description="Emission scopes included"
    )
    data_coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Data coverage percentage"
    )
    methodology: str = Field(
        "", description="Carbon intensity calculation methodology"
    )


class DecarbonizationTrajectory(BaseModel):
    """Decarbonization trajectory data."""

    baseline_year: int = Field(2019, ge=2015, le=2030, description="Baseline year")
    baseline_intensity: Optional[float] = Field(
        None, description="Baseline carbon intensity"
    )
    current_intensity: Optional[float] = Field(
        None, description="Current carbon intensity"
    )
    target_2030: Optional[float] = Field(
        None, description="2030 target intensity"
    )
    target_2050: Optional[float] = Field(
        None, description="2050 target intensity"
    )
    annual_reduction_target_pct: float = Field(
        7.0, ge=0.0, le=100.0,
        description="Annual YoY reduction target (PAB: 7%, CTB: per benchmark)",
    )
    actual_yoy_reduction_pct: Optional[float] = Field(
        None, description="Actual YoY reduction achieved"
    )
    on_track: bool = Field(
        True, description="Whether trajectory is on track"
    )
    trajectory_milestones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Trajectory milestones: {year, target, actual, unit}",
    )
    paris_alignment_status: str = Field(
        "aligned",
        description="Paris alignment status: aligned, at_risk, misaligned",
    )
    scenario_reference: str = Field(
        "IEA Net Zero 2050",
        description="Climate scenario used as reference",
    )


class ExclusionCompliance(BaseModel):
    """Exclusion criteria compliance for CTB/PAB."""

    exclusion_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Exclusion rules: {rule, threshold, compliant, details}",
    )
    controversial_weapons_excluded: bool = Field(
        True, description="Controversial weapons excluded"
    )
    tobacco_excluded: bool = Field(
        True, description="Tobacco manufacturing excluded"
    )
    fossil_fuel_exclusions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Fossil fuel exclusion details: {fuel_type, revenue_threshold, compliant}",
    )
    un_global_compact_violators: bool = Field(
        True, description="UN Global Compact violators excluded"
    )
    overall_compliance: bool = Field(
        True, description="Overall exclusion compliance status"
    )
    non_compliant_holdings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Non-compliant holdings: {name, reason, weight_pct, action}",
    )
    last_review_date: str = Field(
        "", description="Last exclusion compliance review date"
    )


class TrackingErrorAnalysis(BaseModel):
    """Tracking error and performance analysis."""

    tracking_error_annualized: Optional[float] = Field(
        None, description="Annualized tracking error (%)"
    )
    tracking_error_limit: Optional[float] = Field(
        None, description="Tracking error limit (%)"
    )
    active_share: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Active share (%)"
    )
    fund_return_ytd: Optional[float] = Field(
        None, description="Fund return YTD (%)"
    )
    benchmark_return_ytd: Optional[float] = Field(
        None, description="Benchmark return YTD (%)"
    )
    excess_return_ytd: Optional[float] = Field(
        None, description="Excess return YTD (%)"
    )
    fund_return_1y: Optional[float] = Field(
        None, description="Fund return 1Y (%)"
    )
    benchmark_return_1y: Optional[float] = Field(
        None, description="Benchmark return 1Y (%)"
    )
    sector_deviation_top: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top sector deviations: {sector, fund_pct, benchmark_pct, diff}",
    )
    rebalancing_notes: str = Field(
        "", description="Notes on tracking and rebalancing"
    )


class BenchmarkMethodologyData(BaseModel):
    """Complete input data for benchmark methodology report."""

    fund_info: BenchmarkFundInfo
    benchmark_selection: BenchmarkSelection = Field(
        default_factory=BenchmarkSelection
    )
    carbon_intensity: CarbonIntensityComparison = Field(
        default_factory=CarbonIntensityComparison
    )
    decarbonization: DecarbonizationTrajectory = Field(
        default_factory=DecarbonizationTrajectory
    )
    exclusion_compliance: ExclusionCompliance = Field(
        default_factory=ExclusionCompliance
    )
    tracking_error: TrackingErrorAnalysis = Field(
        default_factory=TrackingErrorAnalysis
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class BenchmarkMethodologyTemplate:
    """
    EU Climate Benchmark methodology report template for Article 9 products.

    Generates a comprehensive benchmark analysis covering CTB/PAB selection
    rationale, carbon intensity comparison, decarbonization trajectory,
    exclusion compliance, and tracking error analysis.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = BenchmarkMethodologyTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Paris-Aligned Benchmark" in md or "Climate Transition" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "benchmark_methodology"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize BenchmarkMethodologyTemplate.

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
        Render benchmark methodology report in the specified format.

        Args:
            data: Report data dictionary matching BenchmarkMethodologyData schema.
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
        Render benchmark methodology report as Markdown.

        Args:
            data: Report data dictionary matching BenchmarkMethodologyData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_selection(data))
        sections.append(self._md_section_2_carbon_intensity(data))
        sections.append(self._md_section_3_decarbonization(data))
        sections.append(self._md_section_4_exclusions(data))
        sections.append(self._md_section_5_tracking_error(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render benchmark methodology report as self-contained HTML.

        Args:
            data: Report data dictionary matching BenchmarkMethodologyData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_selection(data))
        sections.append(self._html_section_2_carbon_intensity(data))
        sections.append(self._html_section_3_decarbonization(data))
        sections.append(self._html_section_4_exclusions(data))
        sections.append(self._html_section_5_tracking_error(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 Benchmark Methodology Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render benchmark methodology report as structured JSON.

        Args:
            data: Report data dictionary matching BenchmarkMethodologyData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        fi = data.get("fund_info", {})
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_benchmark_methodology",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "fund_info": fi,
            "benchmark_selection": data.get("benchmark_selection", {}),
            "carbon_intensity": data.get("carbon_intensity", {}),
            "decarbonization": data.get("decarbonization", {}),
            "exclusion_compliance": data.get("exclusion_compliance", {}),
            "tracking_error": data.get("tracking_error", {}),
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
        bm_name = fi.get("benchmark_name", "")
        return (
            f"# Benchmark Methodology Report (SFDR Article 9)\n\n"
            f"**Fund:** {name}\n\n"
            f"**Designated Benchmark:** {bm_name or 'N/A'}\n\n"
            f"**Reporting Date:** {fi.get('reporting_date', '')}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_selection(self, data: Dict[str, Any]) -> str:
        """Section 1: CTB/PAB selection rationale."""
        bs = data.get("benchmark_selection", {})
        bm_type = bs.get("benchmark_type", "PAB")
        rationale = bs.get("selection_rationale", "")
        alignment = bs.get("alignment_with_objective", "")
        regulation = bs.get("eu_regulation_reference", "")
        ctb_criteria = bs.get("ctb_compliance_criteria", [])
        pab_criteria = bs.get("pab_compliance_criteria", [])
        deviation = bs.get("deviation_from_parent", "")
        rebalancing = bs.get("rebalancing_frequency", "quarterly")

        type_full = {
            "PAB": "Paris-Aligned Benchmark (PAB)",
            "CTB": "Climate Transition Benchmark (CTB)",
        }

        lines: List[str] = [
            "## 1. Benchmark Selection Rationale\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Benchmark Type** | {type_full.get(bm_type, bm_type)} |",
            f"| **EU Regulation** | {regulation} |",
            f"| **Rebalancing Frequency** | {rebalancing} |",
            "",
        ]

        if rationale:
            lines.append(f"### Selection Rationale\n\n{rationale}\n")

        if alignment:
            lines.append(f"### Alignment with Sustainable Objective\n\n{alignment}\n")

        criteria = pab_criteria if bm_type == "PAB" else ctb_criteria
        label = "PAB" if bm_type == "PAB" else "CTB"
        if criteria:
            lines.append(f"### {label} Compliance Criteria\n")
            for c in criteria:
                lines.append(f"- {c}")
            lines.append("")

        if deviation:
            lines.append(f"### Deviation from Parent Index\n\n{deviation}\n")

        return "\n".join(lines)

    def _md_section_2_carbon_intensity(self, data: Dict[str, Any]) -> str:
        """Section 2: Carbon intensity comparison."""
        ci = data.get("carbon_intensity", {})
        fund_waci = ci.get("fund_waci")
        bm_waci = ci.get("benchmark_waci")
        parent_waci = ci.get("parent_index_waci")
        reduction = ci.get("reduction_vs_parent")
        fund_cf = ci.get("fund_carbon_footprint")
        bm_cf = ci.get("benchmark_carbon_footprint")
        fund_te = ci.get("fund_total_emissions")
        bm_te = ci.get("benchmark_total_emissions")
        scope = ci.get("scope_coverage", "Scope 1+2")
        coverage = ci.get("data_coverage_pct", 0.0)
        methodology = ci.get("methodology", "")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        lines: List[str] = [
            "## 2. Carbon Intensity Comparison\n",
            f"**Scope Coverage:** {scope} | **Data Coverage:** {coverage:.1f}%\n",
            "### Weighted Average Carbon Intensity (WACI)\n",
            "| Metric | Fund | Benchmark | Parent Index |",
            "|--------|------|-----------|-------------|",
            f"| WACI (tCO2e/M revenue) | {_fmt(fund_waci)} | {_fmt(bm_waci)} | {_fmt(parent_waci)} |",
            "",
        ]

        if reduction is not None:
            lines.append(f"**Reduction vs. Parent Index:** {reduction:.1f}%\n")

        lines.append("### Carbon Footprint\n")
        lines.append("| Metric | Fund | Benchmark |")
        lines.append("|--------|------|-----------|")
        lines.append(
            f"| Carbon Footprint (tCO2e/M invested) | {_fmt(fund_cf)} | {_fmt(bm_cf)} |"
        )
        lines.append(
            f"| Total Financed Emissions (tCO2e) | {_fmt(fund_te)} | {_fmt(bm_te)} |"
        )
        lines.append("")

        # ASCII bar chart
        lines.append("### Visual Comparison\n")
        lines.append("```")
        items = [
            ("Fund WACI", fund_waci or 0),
            ("Benchmark WACI", bm_waci or 0),
            ("Parent WACI", parent_waci or 0),
        ]
        max_val = max((v for _, v in items), default=1) or 1
        for label, val in items:
            bar_len = int((val / max_val) * 40)
            bar = "#" * bar_len
            lines.append(f"  {label:20s} [{bar:<40s}] {val:,.1f}")
        lines.append("```")

        if methodology:
            lines.append(f"\n### Methodology\n\n{methodology}")

        return "\n".join(lines)

    def _md_section_3_decarbonization(self, data: Dict[str, Any]) -> str:
        """Section 3: Decarbonization trajectory."""
        dec = data.get("decarbonization", {})
        baseline_year = dec.get("baseline_year", 2019)
        baseline_int = dec.get("baseline_intensity")
        current_int = dec.get("current_intensity")
        target_2030 = dec.get("target_2030")
        target_2050 = dec.get("target_2050")
        annual_target = dec.get("annual_reduction_target_pct", 7.0)
        actual_yoy = dec.get("actual_yoy_reduction_pct")
        on_track = dec.get("on_track", True)
        milestones = dec.get("trajectory_milestones", [])
        paris_status = dec.get("paris_alignment_status", "aligned")
        scenario = dec.get("scenario_reference", "")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        status_display = {
            "aligned": "ALIGNED",
            "at_risk": "AT RISK",
            "misaligned": "MISALIGNED",
        }

        lines: List[str] = [
            "## 3. Decarbonization Trajectory\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Baseline Year** | {baseline_year} |",
            f"| **Baseline Intensity** | {_fmt(baseline_int)} |",
            f"| **Current Intensity** | {_fmt(current_int)} |",
            f"| **2030 Target** | {_fmt(target_2030)} |",
            f"| **2050 Target** | {_fmt(target_2050)} |",
            f"| **Annual Reduction Target** | {annual_target:.1f}% YoY |",
            f"| **Actual YoY Reduction** | {_fmt(actual_yoy)}% |",
            f"| **On Track** | {'Yes' if on_track else 'No'} |",
            f"| **Paris Alignment** | {status_display.get(paris_status, paris_status.upper())} |",
            f"| **Scenario Reference** | {scenario or 'N/A'} |",
            "",
        ]

        if milestones:
            lines.append("### Trajectory Milestones\n")
            lines.append("| Year | Target | Actual | Unit |")
            lines.append("|------|--------|--------|------|")
            for m in milestones:
                t = m.get("target")
                a = m.get("actual")
                lines.append(
                    f"| {m.get('year', '')} | "
                    f"{_fmt(t)} | "
                    f"{_fmt(a)} | "
                    f"{m.get('unit', '')} |"
                )
            lines.append("")

        # ASCII trajectory chart
        lines.append("### Trajectory Visualization\n")
        lines.append("```")
        points = [(baseline_year, baseline_int or 0)]
        for m in milestones:
            y = m.get("year")
            a = m.get("actual") or m.get("target")
            if y is not None and a is not None:
                points.append((y, a))
        if target_2030 is not None:
            points.append((2030, target_2030))
        if target_2050 is not None:
            points.append((2050, target_2050))
        max_val = max((v for _, v in points), default=1) or 1
        for year, val in points:
            bar_len = int((val / max_val) * 40)
            bar = "#" * bar_len
            lines.append(f"  {year}  [{bar:<40s}] {val:,.1f}")
        lines.append("```")

        return "\n".join(lines)

    def _md_section_4_exclusions(self, data: Dict[str, Any]) -> str:
        """Section 4: Exclusion compliance."""
        exc = data.get("exclusion_compliance", {})
        rules = exc.get("exclusion_rules", [])
        cw = exc.get("controversial_weapons_excluded", True)
        tobacco = exc.get("tobacco_excluded", True)
        fossil = exc.get("fossil_fuel_exclusions", [])
        ungc = exc.get("un_global_compact_violators", True)
        overall = exc.get("overall_compliance", True)
        non_compliant = exc.get("non_compliant_holdings", [])
        review_date = exc.get("last_review_date", "")

        lines: List[str] = [
            "## 4. Exclusion Compliance\n",
            f"**Overall Compliance:** {'PASS' if overall else 'FAIL'}\n",
            f"**Last Review:** {review_date or 'N/A'}\n",
            "### Mandatory Exclusions\n",
            "| Exclusion | Status |",
            "|-----------|--------|",
            f"| Controversial weapons | {'Compliant' if cw else 'Non-compliant'} |",
            f"| Tobacco manufacturing | {'Compliant' if tobacco else 'Non-compliant'} |",
            f"| UN Global Compact violators | {'Compliant' if ungc else 'Non-compliant'} |",
            "",
        ]

        if fossil:
            lines.append("### Fossil Fuel Exclusions\n")
            lines.append("| Fuel Type | Revenue Threshold | Compliant |")
            lines.append("|-----------|-------------------|----------|")
            for f_item in fossil:
                lines.append(
                    f"| {f_item.get('fuel_type', '')} | "
                    f"{f_item.get('revenue_threshold', '')} | "
                    f"{'Yes' if f_item.get('compliant', True) else 'No'} |"
                )
            lines.append("")

        if rules:
            lines.append("### Additional Exclusion Rules\n")
            lines.append("| Rule | Threshold | Compliant | Details |")
            lines.append("|------|-----------|-----------|--------|")
            for r in rules:
                lines.append(
                    f"| {r.get('rule', '')} | "
                    f"{r.get('threshold', '')} | "
                    f"{'Yes' if r.get('compliant', True) else 'No'} | "
                    f"{r.get('details', '')} |"
                )
            lines.append("")

        if non_compliant:
            lines.append("### Non-Compliant Holdings\n")
            lines.append("| Holding | Reason | Weight (%) | Action |")
            lines.append("|---------|--------|-----------|--------|")
            for h in non_compliant:
                lines.append(
                    f"| {h.get('name', '')} | "
                    f"{h.get('reason', '')} | "
                    f"{h.get('weight_pct', 0.0):.2f}% | "
                    f"{h.get('action', '')} |"
                )

        return "\n".join(lines)

    def _md_section_5_tracking_error(self, data: Dict[str, Any]) -> str:
        """Section 5: Tracking error analysis."""
        te = data.get("tracking_error", {})
        te_ann = te.get("tracking_error_annualized")
        te_limit = te.get("tracking_error_limit")
        active_share = te.get("active_share")
        fund_ytd = te.get("fund_return_ytd")
        bm_ytd = te.get("benchmark_return_ytd")
        excess_ytd = te.get("excess_return_ytd")
        fund_1y = te.get("fund_return_1y")
        bm_1y = te.get("benchmark_return_1y")
        sectors = te.get("sector_deviation_top", [])
        notes = te.get("rebalancing_notes", "")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.2f}%" if v is not None else "N/A"

        lines: List[str] = [
            "## 5. Tracking Error & Performance Analysis\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Annualized Tracking Error** | {_fmt(te_ann)} |",
            f"| **Tracking Error Limit** | {_fmt(te_limit)} |",
            f"| **Active Share** | {_fmt(active_share)} |",
            "",
            "### Performance Comparison\n",
            "| Period | Fund | Benchmark | Excess |",
            "|--------|------|-----------|--------|",
            f"| YTD | {_fmt(fund_ytd)} | {_fmt(bm_ytd)} | {_fmt(excess_ytd)} |",
            f"| 1 Year | {_fmt(fund_1y)} | {_fmt(bm_1y)} | N/A |",
            "",
        ]

        if sectors:
            lines.append("### Top Sector Deviations\n")
            lines.append("| Sector | Fund (%) | Benchmark (%) | Deviation |")
            lines.append("|--------|----------|---------------|----------|")
            for s in sectors:
                diff = s.get("diff", 0.0)
                lines.append(
                    f"| {s.get('sector', '')} | "
                    f"{s.get('fund_pct', 0.0):.1f}% | "
                    f"{s.get('benchmark_pct', 0.0):.1f}% | "
                    f"{diff:+.1f}% |"
                )
            lines.append("")

        if notes:
            lines.append(f"### Rebalancing Notes\n\n{notes}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_selection(self, data: Dict[str, Any]) -> str:
        """Build HTML benchmark selection section."""
        bs = data.get("benchmark_selection", {})
        bm_type = bs.get("benchmark_type", "PAB")
        rationale = bs.get("selection_rationale", "")
        alignment = bs.get("alignment_with_objective", "")
        criteria = (
            bs.get("pab_compliance_criteria", [])
            if bm_type == "PAB"
            else bs.get("ctb_compliance_criteria", [])
        )

        type_full = {"PAB": "Paris-Aligned Benchmark", "CTB": "Climate Transition Benchmark"}
        parts: List[str] = [
            '<div class="section"><h2>1. Benchmark Selection Rationale</h2>',
            f'<p><strong>Type:</strong> {_esc(type_full.get(bm_type, bm_type))}</p>',
        ]

        if rationale:
            parts.append(f"<h3>Rationale</h3><p>{_esc(rationale)}</p>")

        if alignment:
            parts.append(f"<h3>Alignment with Objective</h3><p>{_esc(alignment)}</p>")

        if criteria:
            parts.append("<h3>Compliance Criteria</h3><ul>")
            for c in criteria:
                parts.append(f"<li>{_esc(c)}</li>")
            parts.append("</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_carbon_intensity(self, data: Dict[str, Any]) -> str:
        """Build HTML carbon intensity section."""
        ci = data.get("carbon_intensity", {})
        fund_waci = ci.get("fund_waci")
        bm_waci = ci.get("benchmark_waci")
        parent_waci = ci.get("parent_index_waci")

        def _fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "N/A"

        parts: List[str] = [
            '<div class="section"><h2>2. Carbon Intensity Comparison</h2>',
            '<table class="data-table">',
            "<tr><th>Metric</th><th>Fund</th><th>Benchmark</th><th>Parent</th></tr>",
            f"<tr><td>WACI (tCO2e/M revenue)</td><td>{_fmt(fund_waci)}</td>"
            f"<td>{_fmt(bm_waci)}</td><td>{_fmt(parent_waci)}</td></tr>",
            "</table>",
            "</div>",
        ]
        return "".join(parts)

    def _html_section_3_decarbonization(self, data: Dict[str, Any]) -> str:
        """Build HTML decarbonization trajectory section."""
        dec = data.get("decarbonization", {})
        on_track = dec.get("on_track", True)
        paris = dec.get("paris_alignment_status", "aligned")
        milestones = dec.get("trajectory_milestones", [])

        colors = {"aligned": "#27ae60", "at_risk": "#f39c12", "misaligned": "#e74c3c"}
        color = colors.get(paris, "#7f8c8d")

        parts: List[str] = [
            '<div class="section"><h2>3. Decarbonization Trajectory</h2>',
            f'<p>On Track: <strong>{"Yes" if on_track else "No"}</strong> | '
            f'Paris Alignment: <span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:3px;">{_esc(paris.upper())}</span></p>',
        ]

        if milestones:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Year</th><th>Target</th><th>Actual</th><th>Unit</th></tr>")
            for m in milestones:
                t = m.get("target")
                a = m.get("actual")
                t_str = f"{t:,.2f}" if t is not None else "N/A"
                a_str = f"{a:,.2f}" if a is not None else "N/A"
                parts.append(
                    f"<tr><td>{m.get('year', '')}</td><td>{t_str}</td>"
                    f"<td>{a_str}</td><td>{_esc(str(m.get('unit', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_exclusions(self, data: Dict[str, Any]) -> str:
        """Build HTML exclusion compliance section."""
        exc = data.get("exclusion_compliance", {})
        overall = exc.get("overall_compliance", True)
        non_compliant = exc.get("non_compliant_holdings", [])

        status_color = "#27ae60" if overall else "#e74c3c"
        status_text = "PASS" if overall else "FAIL"

        parts: List[str] = [
            '<div class="section"><h2>4. Exclusion Compliance</h2>',
            f'<p>Overall: <span style="background:{status_color};color:white;'
            f'padding:2px 8px;border-radius:3px;font-weight:bold;">{status_text}</span></p>',
        ]

        if non_compliant:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Holding</th><th>Reason</th><th>Weight</th><th>Action</th></tr>")
            for h in non_compliant:
                parts.append(
                    f"<tr><td>{_esc(str(h.get('name', '')))}</td>"
                    f"<td>{_esc(str(h.get('reason', '')))}</td>"
                    f"<td>{h.get('weight_pct', 0.0):.2f}%</td>"
                    f"<td>{_esc(str(h.get('action', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_tracking_error(self, data: Dict[str, Any]) -> str:
        """Build HTML tracking error section."""
        te = data.get("tracking_error", {})
        te_ann = te.get("tracking_error_annualized")
        active_share = te.get("active_share")
        sectors = te.get("sector_deviation_top", [])

        def _fmt(v: Optional[float]) -> str:
            return f"{v:.2f}%" if v is not None else "N/A"

        parts: List[str] = [
            '<div class="section"><h2>5. Tracking Error & Performance</h2>',
            f"<p>Tracking Error: <strong>{_fmt(te_ann)}</strong> | "
            f"Active Share: <strong>{_fmt(active_share)}</strong></p>",
        ]

        if sectors:
            parts.append('<table class="data-table">')
            parts.append("<tr><th>Sector</th><th>Fund</th><th>Benchmark</th><th>Deviation</th></tr>")
            for s in sectors:
                diff = s.get("diff", 0.0)
                parts.append(
                    f"<tr><td>{_esc(str(s.get('sector', '')))}</td>"
                    f"<td>{s.get('fund_pct', 0.0):.1f}%</td>"
                    f"<td>{s.get('benchmark_pct', 0.0):.1f}%</td>"
                    f"<td>{diff:+.1f}%</td></tr>"
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
