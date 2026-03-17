"""
SourcingScenarioAnalysis - CBAM sourcing scenario and what-if analysis template.

This module implements the sourcing scenario analysis report for PACK-005 CBAM
Complete. It generates reports covering current sourcing profiles, multi-scenario
comparisons, Monte Carlo simulation results, sensitivity analysis, supplier
switching impacts, decarbonization ROI calculations, ranked recommendations,
and modeling assumptions.

Example:
    >>> template = SourcingScenarioAnalysis()
    >>> data = SourcingScenarioData(
    ...     current_profile=CurrentProfile(total_suppliers=15, ...),
    ...     scenarios=[Scenario(scenario_id="S1", ...)],
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class SupplierProfile(BaseModel):
    """Individual supplier in current sourcing profile."""

    supplier_name: str = Field("", description="Supplier name")
    country: str = Field("", description="Country of origin")
    product: str = Field("", description="Product/material")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Import volume tonnes")
    emission_intensity_tco2e_per_t: float = Field(0.0, ge=0.0, description="Emission intensity")
    cbam_cost_per_tonne_eur: float = Field(0.0, ge=0.0, description="CBAM cost per tonne")
    share_pct: float = Field(0.0, ge=0.0, le=100.0, description="Share of total sourcing")


class CurrentProfile(BaseModel):
    """Current sourcing profile overview."""

    total_suppliers: int = Field(0, ge=0, description="Total number of suppliers")
    total_volume_tonnes: float = Field(0.0, ge=0.0, description="Total import volume")
    weighted_avg_emission_intensity: float = Field(0.0, ge=0.0, description="Weighted avg emission intensity")
    total_embedded_emissions_tco2e: float = Field(0.0, ge=0.0, description="Total embedded emissions")
    total_cbam_cost_eur: float = Field(0.0, ge=0.0, description="Total annual CBAM cost")
    suppliers: List[SupplierProfile] = Field(default_factory=list, description="Supplier details")


class Scenario(BaseModel):
    """Individual sourcing scenario for comparison."""

    scenario_id: str = Field("", description="Scenario identifier")
    scenario_name: str = Field("", description="Scenario name")
    description: str = Field("", description="Scenario description")
    total_volume_tonnes: float = Field(0.0, ge=0.0, description="Total volume")
    weighted_avg_emission_intensity: float = Field(0.0, ge=0.0, description="Weighted avg intensity")
    total_embedded_emissions_tco2e: float = Field(0.0, ge=0.0, description="Total emissions")
    total_cbam_cost_eur: float = Field(0.0, ge=0.0, description="Total CBAM cost")
    logistics_cost_delta_eur: float = Field(0.0, description="Logistics cost change")
    product_cost_delta_eur: float = Field(0.0, description="Product cost change")
    risk_score: float = Field(0.0, ge=0.0, le=100.0, description="Risk score 0-100")
    feasibility: str = Field("medium", description="Feasibility: low, medium, high")


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation results."""

    num_simulations: int = Field(10000, ge=100, description="Number of simulations")
    p5_cost_eur: float = Field(0.0, ge=0.0, description="5th percentile cost")
    p25_cost_eur: float = Field(0.0, ge=0.0, description="25th percentile cost")
    p50_cost_eur: float = Field(0.0, ge=0.0, description="Median cost")
    p75_cost_eur: float = Field(0.0, ge=0.0, description="75th percentile cost")
    p95_cost_eur: float = Field(0.0, ge=0.0, description="95th percentile cost")
    mean_cost_eur: float = Field(0.0, ge=0.0, description="Mean cost")
    std_dev_eur: float = Field(0.0, ge=0.0, description="Standard deviation")
    histogram_buckets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Histogram bucket data: range_label, count, pct"
    )


class SensitivityVariable(BaseModel):
    """Sensitivity analysis variable."""

    variable_name: str = Field("", description="Variable name")
    base_value: float = Field(0.0, description="Base case value")
    low_value: float = Field(0.0, description="Low case value")
    high_value: float = Field(0.0, description="High case value")
    low_impact_eur: float = Field(0.0, description="Cost impact at low value")
    high_impact_eur: float = Field(0.0, description="Cost impact at high value")
    impact_range_eur: float = Field(0.0, ge=0.0, description="Total impact range")


class SupplierSwitch(BaseModel):
    """Supplier switching what-if analysis."""

    from_supplier: str = Field("", description="Current supplier")
    to_supplier: str = Field("", description="Replacement supplier")
    from_country: str = Field("", description="Current country")
    to_country: str = Field("", description="New country")
    volume_tonnes: float = Field(0.0, ge=0.0, description="Volume to switch")
    emission_delta_tco2e: float = Field(0.0, description="Emission change")
    cost_delta_eur: float = Field(0.0, description="CBAM cost change")
    logistics_delta_eur: float = Field(0.0, description="Logistics cost change")
    net_impact_eur: float = Field(0.0, description="Net financial impact")


class DecarbonizationROI(BaseModel):
    """Decarbonization return on investment analysis."""

    supplier_name: str = Field("", description="Supplier name")
    current_intensity_tco2e_per_t: float = Field(0.0, ge=0.0, description="Current intensity")
    target_intensity_tco2e_per_t: float = Field(0.0, ge=0.0, description="Target intensity")
    reduction_pct: float = Field(0.0, ge=0.0, le=100.0, description="Reduction percentage")
    annual_cbam_savings_eur: float = Field(0.0, ge=0.0, description="Annual CBAM savings")
    estimated_investment_eur: float = Field(0.0, ge=0.0, description="Investment needed")
    payback_years: float = Field(0.0, ge=0.0, description="Payback period in years")
    irr_pct: Optional[float] = Field(None, description="Internal rate of return")


class Recommendation(BaseModel):
    """Ranked recommendation."""

    rank: int = Field(0, ge=1, description="Priority rank")
    action: str = Field("", description="Recommended action")
    expected_savings_eur: float = Field(0.0, description="Expected annual savings")
    implementation_complexity: str = Field("medium", description="low, medium, high")
    timeline_months: int = Field(0, ge=0, description="Implementation timeline")
    confidence: str = Field("medium", description="low, medium, high")


class Assumption(BaseModel):
    """Modeling assumption."""

    assumption_id: str = Field("", description="Assumption identifier")
    category: str = Field("", description="Assumption category")
    description: str = Field("", description="Assumption description")
    value: str = Field("", description="Assumed value")
    source: str = Field("", description="Data source")
    sensitivity: str = Field("low", description="Sensitivity: low, medium, high")


class SourcingScenarioData(BaseModel):
    """Complete input data for sourcing scenario analysis."""

    current_profile: CurrentProfile = Field(default_factory=CurrentProfile)
    scenarios: List[Scenario] = Field(default_factory=list)
    monte_carlo: MonteCarloResult = Field(default_factory=MonteCarloResult)
    sensitivity_variables: List[SensitivityVariable] = Field(default_factory=list)
    supplier_switches: List[SupplierSwitch] = Field(default_factory=list)
    decarbonization_roi: List[DecarbonizationROI] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class SourcingScenarioAnalysis:
    """
    CBAM sourcing scenario analysis template.

    Generates scenario analysis reports for CBAM sourcing optimization,
    including multi-scenario comparisons, Monte Carlo simulations,
    sensitivity analysis, supplier switching impacts, decarbonization ROI,
    and ranked recommendations.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = SourcingScenarioAnalysis()
        >>> md = template.render_markdown(data)
        >>> assert "Scenario Comparison" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "sourcing_scenario_analysis"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize SourcingScenarioAnalysis.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the sourcing scenario analysis as Markdown.

        Args:
            data: Report data dictionary matching SourcingScenarioData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_current_profile(data))
        sections.append(self._md_scenario_comparison(data))
        sections.append(self._md_monte_carlo(data))
        sections.append(self._md_sensitivity(data))
        sections.append(self._md_supplier_switching(data))
        sections.append(self._md_decarbonization_roi(data))
        sections.append(self._md_recommendations(data))
        sections.append(self._md_assumptions(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the sourcing scenario analysis as self-contained HTML.

        Args:
            data: Report data dictionary matching SourcingScenarioData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_current_profile(data))
        sections.append(self._html_scenario_comparison(data))
        sections.append(self._html_monte_carlo(data))
        sections.append(self._html_sensitivity(data))
        sections.append(self._html_supplier_switching(data))
        sections.append(self._html_decarbonization_roi(data))
        sections.append(self._html_recommendations(data))
        sections.append(self._html_assumptions(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Sourcing Scenario Analysis",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the sourcing scenario analysis as structured JSON.

        Args:
            data: Report data dictionary matching SourcingScenarioData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_sourcing_scenario_analysis",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "current_profile": self._json_current_profile(data),
            "scenarios": self._json_scenarios(data),
            "monte_carlo": self._json_monte_carlo(data),
            "sensitivity_analysis": self._json_sensitivity(data),
            "supplier_switches": self._json_supplier_switches(data),
            "decarbonization_roi": self._json_decarbonization_roi(data),
            "recommendations": self._json_recommendations(data),
            "assumptions": self._json_assumptions(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown report header."""
        return (
            "# CBAM Sourcing Scenario Analysis\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_current_profile(self, data: Dict[str, Any]) -> str:
        """Build Markdown current sourcing profile section."""
        cp = data.get("current_profile", {})
        cur = self._currency()
        suppliers = cp.get("suppliers", [])

        summary = (
            "## 1. Current Sourcing Profile\n\n"
            "### Overview\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Total Suppliers | {cp.get('total_suppliers', 0)} |\n"
            f"| Total Volume | {self._fmt_num(cp.get('total_volume_tonnes', 0.0))} tonnes |\n"
            f"| Weighted Avg Emission Intensity | {self._fmt_num(cp.get('weighted_avg_emission_intensity', 0.0), 4)} tCO2e/t |\n"
            f"| Total Embedded Emissions | {self._fmt_num(cp.get('total_embedded_emissions_tco2e', 0.0))} tCO2e |\n"
            f"| Total CBAM Cost | {self._fmt_cur(cp.get('total_cbam_cost_eur', 0.0), cur)} |"
        )

        if not suppliers:
            return summary

        detail = (
            "\n\n### Supplier Details\n\n"
            "| Supplier | Country | Product | Volume (t) | Intensity (tCO2e/t) | CBAM Cost/t | Share |\n"
            "|----------|---------|---------|------------|---------------------|-------------|-------|\n"
        )

        rows: List[str] = []
        for s in suppliers:
            rows.append(
                f"| {s.get('supplier_name', '')} | "
                f"{s.get('country', '')} | "
                f"{s.get('product', '')} | "
                f"{self._fmt_num(s.get('volume_tonnes', 0.0))} | "
                f"{self._fmt_num(s.get('emission_intensity_tco2e_per_t', 0.0), 4)} | "
                f"{self._fmt_cur(s.get('cbam_cost_per_tonne_eur', 0.0), cur)} | "
                f"{s.get('share_pct', 0.0):.1f}% |"
            )

        return summary + detail + "\n".join(rows)

    def _md_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Build Markdown scenario comparison table section."""
        scenarios = data.get("scenarios", [])
        cur = self._currency()

        header = (
            "## 2. Scenario Comparison\n\n"
            "| Scenario | Volume (t) | Avg Intensity | Emissions (tCO2e) | CBAM Cost | "
            "Logistics Delta | Product Delta | Risk | Feasibility |\n"
            "|----------|-----------|---------------|-------------------|-----------|"
            "-----------------|---------------|------|-------------|\n"
        )

        rows: List[str] = []
        for sc in scenarios:
            log_delta = sc.get("logistics_cost_delta_eur", 0.0)
            prod_delta = sc.get("product_cost_delta_eur", 0.0)
            log_sign = "+" if log_delta >= 0 else ""
            prod_sign = "+" if prod_delta >= 0 else ""

            rows.append(
                f"| **{sc.get('scenario_name', '')}** | "
                f"{self._fmt_num(sc.get('total_volume_tonnes', 0.0))} | "
                f"{self._fmt_num(sc.get('weighted_avg_emission_intensity', 0.0), 4)} | "
                f"{self._fmt_num(sc.get('total_embedded_emissions_tco2e', 0.0))} | "
                f"{self._fmt_cur(sc.get('total_cbam_cost_eur', 0.0), cur)} | "
                f"{log_sign}{self._fmt_cur(log_delta, cur)} | "
                f"{prod_sign}{self._fmt_cur(prod_delta, cur)} | "
                f"{sc.get('risk_score', 0.0):.0f}/100 | "
                f"{sc.get('feasibility', 'medium').upper()} |"
            )

        if not rows:
            return header + "| *No scenarios defined* | | | | | | | | |"

        return header + "\n".join(rows)

    def _md_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Build Markdown Monte Carlo results section with text histogram."""
        mc = data.get("monte_carlo", {})
        cur = self._currency()
        buckets = mc.get("histogram_buckets", [])

        summary = (
            "## 3. Monte Carlo Simulation Results\n\n"
            f"**Simulations:** {self._fmt_int(mc.get('num_simulations', 0))}\n\n"
            "### Cost Distribution\n\n"
            "| Percentile | Cost |\n"
            "|------------|------|\n"
            f"| P5 (optimistic) | {self._fmt_cur(mc.get('p5_cost_eur', 0.0), cur)} |\n"
            f"| P25 | {self._fmt_cur(mc.get('p25_cost_eur', 0.0), cur)} |\n"
            f"| **P50 (median)** | **{self._fmt_cur(mc.get('p50_cost_eur', 0.0), cur)}** |\n"
            f"| P75 | {self._fmt_cur(mc.get('p75_cost_eur', 0.0), cur)} |\n"
            f"| P95 (pessimistic) | {self._fmt_cur(mc.get('p95_cost_eur', 0.0), cur)} |\n"
            f"| Mean | {self._fmt_cur(mc.get('mean_cost_eur', 0.0), cur)} |\n"
            f"| Std Dev | {self._fmt_cur(mc.get('std_dev_eur', 0.0), cur)} |"
        )

        if not buckets:
            return summary

        # Text-based histogram
        max_count = max((b.get("count", 0) for b in buckets), default=1)
        max_count = max(max_count, 1)
        bar_max_width = 40

        histogram = "\n\n### Cost Distribution Histogram\n\n```\n"
        for b in buckets:
            label = b.get("range_label", "")
            count = b.get("count", 0)
            pct = b.get("pct", 0.0)
            bar_width = int(count / max_count * bar_max_width)
            bar = "#" * bar_width
            histogram += f"{label:>20s} | {bar:<{bar_max_width}s} {pct:5.1f}%\n"
        histogram += "```"

        return summary + histogram

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        """Build Markdown sensitivity analysis (tornado diagram data)."""
        variables = data.get("sensitivity_variables", [])
        cur = self._currency()

        # Sort by impact range descending (tornado order)
        sorted_vars = sorted(
            variables,
            key=lambda v: v.get("impact_range_eur", 0.0),
            reverse=True,
        )

        header = (
            "## 4. Sensitivity Analysis\n\n"
            "Variables ranked by impact on total CBAM cost (tornado diagram).\n\n"
            "| Variable | Base Value | Low Impact | High Impact | Impact Range |\n"
            "|----------|-----------|------------|-------------|-------------|\n"
        )

        rows: List[str] = []
        for v in sorted_vars:
            low_imp = v.get("low_impact_eur", 0.0)
            high_imp = v.get("high_impact_eur", 0.0)
            low_sign = "+" if low_imp >= 0 else ""
            high_sign = "+" if high_imp >= 0 else ""
            rows.append(
                f"| {v.get('variable_name', '')} | "
                f"{v.get('base_value', 0.0)} | "
                f"{low_sign}{self._fmt_cur(low_imp, cur)} | "
                f"{high_sign}{self._fmt_cur(high_imp, cur)} | "
                f"{self._fmt_cur(v.get('impact_range_eur', 0.0), cur)} |"
            )

        if not rows:
            return header + "| *No variables analyzed* | | | | |"

        return header + "\n".join(rows)

    def _md_supplier_switching(self, data: Dict[str, Any]) -> str:
        """Build Markdown supplier switching impact section."""
        switches = data.get("supplier_switches", [])
        cur = self._currency()

        header = (
            "## 5. Supplier Switching Impact\n\n"
            "| From | To | Volume (t) | Emission Delta | Cost Delta | "
            "Logistics Delta | Net Impact |\n"
            "|------|----|-----------|----------------|------------|"
            "-----------------|------------|\n"
        )

        rows: List[str] = []
        for sw in switches:
            em_delta = sw.get("emission_delta_tco2e", 0.0)
            cost_delta = sw.get("cost_delta_eur", 0.0)
            log_delta = sw.get("logistics_delta_eur", 0.0)
            net = sw.get("net_impact_eur", 0.0)

            em_sign = "+" if em_delta >= 0 else ""
            cost_sign = "+" if cost_delta >= 0 else ""
            log_sign = "+" if log_delta >= 0 else ""
            net_sign = "+" if net >= 0 else ""

            from_label = f"{sw.get('from_supplier', '')} ({sw.get('from_country', '')})"
            to_label = f"{sw.get('to_supplier', '')} ({sw.get('to_country', '')})"

            rows.append(
                f"| {from_label} | {to_label} | "
                f"{self._fmt_num(sw.get('volume_tonnes', 0.0))} | "
                f"{em_sign}{self._fmt_num(em_delta)} tCO2e | "
                f"{cost_sign}{self._fmt_cur(cost_delta, cur)} | "
                f"{log_sign}{self._fmt_cur(log_delta, cur)} | "
                f"{net_sign}{self._fmt_cur(net, cur)} |"
            )

        if not rows:
            return header + "| *No switches analyzed* | | | | | | |"

        return header + "\n".join(rows)

    def _md_decarbonization_roi(self, data: Dict[str, Any]) -> str:
        """Build Markdown decarbonization ROI section."""
        roi_items = data.get("decarbonization_roi", [])
        cur = self._currency()

        header = (
            "## 6. Decarbonization ROI\n\n"
            "| Supplier | Current (tCO2e/t) | Target (tCO2e/t) | Reduction | "
            "Annual Savings | Investment | Payback (yr) | IRR |\n"
            "|----------|-------------------|------------------|-----------|"
            "----------------|------------|-------------|-----|\n"
        )

        rows: List[str] = []
        for r in roi_items:
            irr = r.get("irr_pct")
            irr_str = f"{irr:.1f}%" if irr is not None else "N/A"
            rows.append(
                f"| {r.get('supplier_name', '')} | "
                f"{self._fmt_num(r.get('current_intensity_tco2e_per_t', 0.0), 4)} | "
                f"{self._fmt_num(r.get('target_intensity_tco2e_per_t', 0.0), 4)} | "
                f"{r.get('reduction_pct', 0.0):.1f}% | "
                f"{self._fmt_cur(r.get('annual_cbam_savings_eur', 0.0), cur)} | "
                f"{self._fmt_cur(r.get('estimated_investment_eur', 0.0), cur)} | "
                f"{r.get('payback_years', 0.0):.1f} | "
                f"{irr_str} |"
            )

        if not rows:
            return header + "| *No ROI analyses* | | | | | | | |"

        return header + "\n".join(rows)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Build Markdown recommendations section."""
        recs = data.get("recommendations", [])
        cur = self._currency()

        sorted_recs = sorted(recs, key=lambda r: r.get("rank", 0))

        header = "## 7. Recommendations\n\n"
        rows_text: List[str] = []

        for r in sorted_recs:
            savings = r.get("expected_savings_eur", 0.0)
            savings_sign = "+" if savings >= 0 else ""
            rows_text.append(
                f"### {r.get('rank', 0)}. {r.get('action', '')}\n\n"
                f"- **Expected Savings:** {savings_sign}{self._fmt_cur(savings, cur)}/year\n"
                f"- **Complexity:** {r.get('implementation_complexity', 'medium').upper()}\n"
                f"- **Timeline:** {r.get('timeline_months', 0)} months\n"
                f"- **Confidence:** {r.get('confidence', 'medium').upper()}"
            )

        if not rows_text:
            return header + "*No recommendations generated.*"

        return header + "\n\n".join(rows_text)

    def _md_assumptions(self, data: Dict[str, Any]) -> str:
        """Build Markdown assumptions section."""
        assumptions = data.get("assumptions", [])

        header = (
            "## 8. Assumptions\n\n"
            "All modeling assumptions listed for transparency and reproducibility.\n\n"
            "| ID | Category | Description | Value | Source | Sensitivity |\n"
            "|----|----------|-------------|-------|--------|-------------|\n"
        )

        rows: List[str] = []
        for a in assumptions:
            rows.append(
                f"| {a.get('assumption_id', '')} | "
                f"{a.get('category', '')} | "
                f"{a.get('description', '')} | "
                f"{a.get('value', '')} | "
                f"{a.get('source', '')} | "
                f"{a.get('sensitivity', 'low').upper()} |"
            )

        if not rows:
            return header + "| *No assumptions documented* | | | | | |"

        return header + "\n".join(rows)

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Pack: {self.PACK_ID}*\n\n"
            f"*Provenance Hash: `{provenance_hash}`*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML report header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Sourcing Scenario Analysis</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_current_profile(self, data: Dict[str, Any]) -> str:
        """Build HTML current sourcing profile section."""
        cp = data.get("current_profile", {})
        cur = self._currency()
        suppliers = cp.get("suppliers", [])

        kpis = (
            f'<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Suppliers</div>'
            f'<div class="kpi-value">{cp.get("total_suppliers", 0)}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Volume</div>'
            f'<div class="kpi-value">{self._fmt_num(cp.get("total_volume_tonnes", 0.0))}</div>'
            f'<div class="kpi-unit">tonnes</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Emissions</div>'
            f'<div class="kpi-value">{self._fmt_num(cp.get("total_embedded_emissions_tco2e", 0.0))}</div>'
            f'<div class="kpi-unit">tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total CBAM Cost</div>'
            f'<div class="kpi-value">{self._fmt_cur(cp.get("total_cbam_cost_eur", 0.0), cur)}</div></div>'
            f'</div>'
        )

        rows_html = ""
        for s in suppliers:
            rows_html += (
                f'<tr><td>{s.get("supplier_name", "")}</td>'
                f'<td>{s.get("country", "")}</td>'
                f'<td>{s.get("product", "")}</td>'
                f'<td class="num">{self._fmt_num(s.get("volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(s.get("emission_intensity_tco2e_per_t", 0.0), 4)}</td>'
                f'<td class="num">{self._fmt_cur(s.get("cbam_cost_per_tonne_eur", 0.0), cur)}</td>'
                f'<td class="num">{s.get("share_pct", 0.0):.1f}%</td></tr>'
            )

        table = ""
        if rows_html:
            table = (
                '<h3>Supplier Details</h3>'
                '<table><thead><tr>'
                '<th>Supplier</th><th>Country</th><th>Product</th>'
                '<th>Volume (t)</th><th>Intensity</th><th>CBAM Cost/t</th><th>Share</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )

        return (
            f'<div class="section"><h2>1. Current Sourcing Profile</h2>'
            f'{kpis}{table}</div>'
        )

    def _html_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML scenario comparison section."""
        scenarios = data.get("scenarios", [])
        cur = self._currency()

        rows_html = ""
        for sc in scenarios:
            risk = sc.get("risk_score", 0.0)
            risk_color = "#2ecc71" if risk < 40 else "#f39c12" if risk < 70 else "#e74c3c"
            feasibility = sc.get("feasibility", "medium").upper()

            rows_html += (
                f'<tr><td><strong>{sc.get("scenario_name", "")}</strong><br>'
                f'<small style="color:#7f8c8d">{sc.get("description", "")}</small></td>'
                f'<td class="num">{self._fmt_num(sc.get("total_volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(sc.get("weighted_avg_emission_intensity", 0.0), 4)}</td>'
                f'<td class="num">{self._fmt_num(sc.get("total_embedded_emissions_tco2e", 0.0))}</td>'
                f'<td class="num">{self._fmt_cur(sc.get("total_cbam_cost_eur", 0.0), cur)}</td>'
                f'<td class="num" style="color:{risk_color}">{risk:.0f}/100</td>'
                f'<td>{feasibility}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No scenarios defined</em></td></tr>'

        return (
            '<div class="section"><h2>2. Scenario Comparison</h2>'
            '<table><thead><tr>'
            '<th>Scenario</th><th>Volume (t)</th><th>Avg Intensity</th>'
            '<th>Emissions (tCO2e)</th><th>CBAM Cost</th>'
            '<th>Risk</th><th>Feasibility</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Build HTML Monte Carlo results section."""
        mc = data.get("monte_carlo", {})
        cur = self._currency()
        buckets = mc.get("histogram_buckets", [])

        percentiles = (
            f'<div class="kpi-grid">'
            f'<div class="kpi-card" style="border-top:3px solid #2ecc71">'
            f'<div class="kpi-label">P5 (Optimistic)</div>'
            f'<div class="kpi-value" style="font-size:20px">'
            f'{self._fmt_cur(mc.get("p5_cost_eur", 0.0), cur)}</div></div>'
            f'<div class="kpi-card" style="border-top:3px solid #3498db">'
            f'<div class="kpi-label">P50 (Median)</div>'
            f'<div class="kpi-value" style="font-size:20px">'
            f'{self._fmt_cur(mc.get("p50_cost_eur", 0.0), cur)}</div></div>'
            f'<div class="kpi-card" style="border-top:3px solid #e74c3c">'
            f'<div class="kpi-label">P95 (Pessimistic)</div>'
            f'<div class="kpi-value" style="font-size:20px">'
            f'{self._fmt_cur(mc.get("p95_cost_eur", 0.0), cur)}</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Std Dev</div>'
            f'<div class="kpi-value" style="font-size:20px">'
            f'{self._fmt_cur(mc.get("std_dev_eur", 0.0), cur)}</div></div>'
            f'</div>'
        )

        histogram_html = ""
        if buckets:
            max_count = max((b.get("count", 0) for b in buckets), default=1)
            max_count = max(max_count, 1)

            bars = ""
            for b in buckets:
                count = b.get("count", 0)
                pct = b.get("pct", 0.0)
                width = count / max_count * 100
                bars += (
                    f'<div style="display:flex;align-items:center;margin-bottom:4px">'
                    f'<div style="width:160px;font-size:12px;text-align:right;'
                    f'padding-right:8px;color:#7f8c8d">{b.get("range_label", "")}</div>'
                    f'<div style="flex:1;background:#ecf0f1;border-radius:3px;height:20px">'
                    f'<div style="width:{width:.0f}%;background:#3498db;height:100%;'
                    f'border-radius:3px"></div></div>'
                    f'<div style="width:60px;font-size:12px;text-align:right;'
                    f'padding-left:8px">{pct:.1f}%</div></div>'
                )

            histogram_html = f'<h3>Cost Distribution</h3><div>{bars}</div>'

        return (
            f'<div class="section"><h2>3. Monte Carlo Simulation Results</h2>'
            f'<p>Simulations: <strong>{self._fmt_int(mc.get("num_simulations", 0))}</strong></p>'
            f'{percentiles}{histogram_html}</div>'
        )

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        """Build HTML sensitivity analysis section."""
        variables = data.get("sensitivity_variables", [])
        cur = self._currency()

        sorted_vars = sorted(
            variables,
            key=lambda v: v.get("impact_range_eur", 0.0),
            reverse=True,
        )

        max_range = max(
            (v.get("impact_range_eur", 0.0) for v in sorted_vars),
            default=1.0,
        )
        max_range = max(max_range, 1.0)

        rows_html = ""
        for v in sorted_vars:
            impact_range = v.get("impact_range_eur", 0.0)
            bar_width = impact_range / max_range * 100

            rows_html += (
                f'<tr><td><strong>{v.get("variable_name", "")}</strong></td>'
                f'<td class="num">{v.get("base_value", 0.0)}</td>'
                f'<td class="num">{self._fmt_cur(v.get("low_impact_eur", 0.0), cur)}</td>'
                f'<td class="num">{self._fmt_cur(v.get("high_impact_eur", 0.0), cur)}</td>'
                f'<td class="num">{self._fmt_cur(impact_range, cur)}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{bar_width:.0f}%;background:#e74c3c"></div></div></td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No variables analyzed</em></td></tr>'

        return (
            '<div class="section"><h2>4. Sensitivity Analysis</h2>'
            '<p>Variables ranked by impact on total CBAM cost (tornado diagram).</p>'
            '<table><thead><tr>'
            '<th>Variable</th><th>Base</th><th>Low Impact</th>'
            '<th>High Impact</th><th>Range</th><th>Impact</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_supplier_switching(self, data: Dict[str, Any]) -> str:
        """Build HTML supplier switching impact section."""
        switches = data.get("supplier_switches", [])
        cur = self._currency()

        rows_html = ""
        for sw in switches:
            net = sw.get("net_impact_eur", 0.0)
            net_color = "#2ecc71" if net < 0 else "#e74c3c"
            net_sign = "+" if net >= 0 else ""

            rows_html += (
                f'<tr><td>{sw.get("from_supplier", "")} ({sw.get("from_country", "")})</td>'
                f'<td>{sw.get("to_supplier", "")} ({sw.get("to_country", "")})</td>'
                f'<td class="num">{self._fmt_num(sw.get("volume_tonnes", 0.0))}</td>'
                f'<td class="num">{self._fmt_num(sw.get("emission_delta_tco2e", 0.0))} tCO2e</td>'
                f'<td class="num">{self._fmt_cur(sw.get("cost_delta_eur", 0.0), cur)}</td>'
                f'<td class="num" style="color:{net_color};font-weight:bold">'
                f'{net_sign}{self._fmt_cur(net, cur)}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No switches analyzed</em></td></tr>'

        return (
            '<div class="section"><h2>5. Supplier Switching Impact</h2>'
            '<table><thead><tr>'
            '<th>From</th><th>To</th><th>Volume (t)</th>'
            '<th>Emission Delta</th><th>Cost Delta</th><th>Net Impact</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_decarbonization_roi(self, data: Dict[str, Any]) -> str:
        """Build HTML decarbonization ROI section."""
        roi_items = data.get("decarbonization_roi", [])
        cur = self._currency()

        rows_html = ""
        for r in roi_items:
            payback = r.get("payback_years", 0.0)
            payback_color = "#2ecc71" if payback < 3 else "#f39c12" if payback < 5 else "#e74c3c"
            irr = r.get("irr_pct")
            irr_str = f"{irr:.1f}%" if irr is not None else "N/A"

            rows_html += (
                f'<tr><td>{r.get("supplier_name", "")}</td>'
                f'<td class="num">{self._fmt_num(r.get("current_intensity_tco2e_per_t", 0.0), 4)}</td>'
                f'<td class="num">{self._fmt_num(r.get("target_intensity_tco2e_per_t", 0.0), 4)}</td>'
                f'<td class="num">{r.get("reduction_pct", 0.0):.1f}%</td>'
                f'<td class="num">{self._fmt_cur(r.get("annual_cbam_savings_eur", 0.0), cur)}</td>'
                f'<td class="num">{self._fmt_cur(r.get("estimated_investment_eur", 0.0), cur)}</td>'
                f'<td class="num" style="color:{payback_color}">{payback:.1f}</td>'
                f'<td class="num">{irr_str}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="8"><em>No ROI analyses</em></td></tr>'

        return (
            '<div class="section"><h2>6. Decarbonization ROI</h2>'
            '<table><thead><tr>'
            '<th>Supplier</th><th>Current (tCO2e/t)</th><th>Target</th>'
            '<th>Reduction</th><th>Annual Savings</th><th>Investment</th>'
            '<th>Payback (yr)</th><th>IRR</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Build HTML recommendations section."""
        recs = data.get("recommendations", [])
        cur = self._currency()

        sorted_recs = sorted(recs, key=lambda r: r.get("rank", 0))

        cards = ""
        complexity_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

        for r in sorted_recs:
            complexity = r.get("implementation_complexity", "medium")
            color = complexity_colors.get(complexity, "#95a5a6")
            savings = r.get("expected_savings_eur", 0.0)
            savings_sign = "+" if savings >= 0 else ""

            cards += (
                f'<div style="background:#f8f9fa;padding:16px;border-radius:8px;'
                f'border-left:4px solid {color};margin-bottom:12px">'
                f'<h3 style="margin:0 0 8px 0">#{r.get("rank", 0)}. {r.get("action", "")}</h3>'
                f'<div style="display:flex;gap:24px;font-size:14px">'
                f'<div><strong>Savings:</strong> {savings_sign}{self._fmt_cur(savings, cur)}/yr</div>'
                f'<div><strong>Complexity:</strong> {complexity.upper()}</div>'
                f'<div><strong>Timeline:</strong> {r.get("timeline_months", 0)} months</div>'
                f'<div><strong>Confidence:</strong> {r.get("confidence", "medium").upper()}</div>'
                f'</div></div>'
            )

        if not cards:
            cards = '<p><em>No recommendations generated.</em></p>'

        return f'<div class="section"><h2>7. Recommendations</h2>{cards}</div>'

    def _html_assumptions(self, data: Dict[str, Any]) -> str:
        """Build HTML assumptions section."""
        assumptions = data.get("assumptions", [])

        rows_html = ""
        sensitivity_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

        for a in assumptions:
            sensitivity = a.get("sensitivity", "low")
            color = sensitivity_colors.get(sensitivity, "#95a5a6")
            rows_html += (
                f'<tr><td>{a.get("assumption_id", "")}</td>'
                f'<td>{a.get("category", "")}</td>'
                f'<td>{a.get("description", "")}</td>'
                f'<td>{a.get("value", "")}</td>'
                f'<td>{a.get("source", "")}</td>'
                f'<td style="color:{color};font-weight:bold">{sensitivity.upper()}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No assumptions documented</em></td></tr>'

        return (
            '<div class="section"><h2>8. Assumptions</h2>'
            '<p>All modeling assumptions listed for transparency and reproducibility.</p>'
            '<table><thead><tr>'
            '<th>ID</th><th>Category</th><th>Description</th>'
            '<th>Value</th><th>Source</th><th>Sensitivity</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_current_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON current profile."""
        cp = data.get("current_profile", {})
        return {
            "total_suppliers": cp.get("total_suppliers", 0),
            "total_volume_tonnes": round(cp.get("total_volume_tonnes", 0.0), 2),
            "weighted_avg_emission_intensity": round(
                cp.get("weighted_avg_emission_intensity", 0.0), 4
            ),
            "total_embedded_emissions_tco2e": round(
                cp.get("total_embedded_emissions_tco2e", 0.0), 2
            ),
            "total_cbam_cost_eur": round(cp.get("total_cbam_cost_eur", 0.0), 2),
            "suppliers": [
                {
                    "supplier_name": s.get("supplier_name", ""),
                    "country": s.get("country", ""),
                    "product": s.get("product", ""),
                    "volume_tonnes": round(s.get("volume_tonnes", 0.0), 2),
                    "emission_intensity_tco2e_per_t": round(
                        s.get("emission_intensity_tco2e_per_t", 0.0), 4
                    ),
                    "cbam_cost_per_tonne_eur": round(s.get("cbam_cost_per_tonne_eur", 0.0), 2),
                    "share_pct": round(s.get("share_pct", 0.0), 2),
                }
                for s in cp.get("suppliers", [])
            ],
        }

    def _json_scenarios(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON scenarios."""
        return [
            {
                "scenario_id": sc.get("scenario_id", ""),
                "scenario_name": sc.get("scenario_name", ""),
                "description": sc.get("description", ""),
                "total_volume_tonnes": round(sc.get("total_volume_tonnes", 0.0), 2),
                "weighted_avg_emission_intensity": round(
                    sc.get("weighted_avg_emission_intensity", 0.0), 4
                ),
                "total_embedded_emissions_tco2e": round(
                    sc.get("total_embedded_emissions_tco2e", 0.0), 2
                ),
                "total_cbam_cost_eur": round(sc.get("total_cbam_cost_eur", 0.0), 2),
                "logistics_cost_delta_eur": round(sc.get("logistics_cost_delta_eur", 0.0), 2),
                "product_cost_delta_eur": round(sc.get("product_cost_delta_eur", 0.0), 2),
                "risk_score": round(sc.get("risk_score", 0.0), 1),
                "feasibility": sc.get("feasibility", "medium"),
            }
            for sc in data.get("scenarios", [])
        ]

    def _json_monte_carlo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON Monte Carlo results."""
        mc = data.get("monte_carlo", {})
        return {
            "num_simulations": mc.get("num_simulations", 0),
            "p5_cost_eur": round(mc.get("p5_cost_eur", 0.0), 2),
            "p25_cost_eur": round(mc.get("p25_cost_eur", 0.0), 2),
            "p50_cost_eur": round(mc.get("p50_cost_eur", 0.0), 2),
            "p75_cost_eur": round(mc.get("p75_cost_eur", 0.0), 2),
            "p95_cost_eur": round(mc.get("p95_cost_eur", 0.0), 2),
            "mean_cost_eur": round(mc.get("mean_cost_eur", 0.0), 2),
            "std_dev_eur": round(mc.get("std_dev_eur", 0.0), 2),
            "histogram_buckets": mc.get("histogram_buckets", []),
        }

    def _json_sensitivity(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON sensitivity analysis."""
        return [
            {
                "variable_name": v.get("variable_name", ""),
                "base_value": v.get("base_value", 0.0),
                "low_value": v.get("low_value", 0.0),
                "high_value": v.get("high_value", 0.0),
                "low_impact_eur": round(v.get("low_impact_eur", 0.0), 2),
                "high_impact_eur": round(v.get("high_impact_eur", 0.0), 2),
                "impact_range_eur": round(v.get("impact_range_eur", 0.0), 2),
            }
            for v in data.get("sensitivity_variables", [])
        ]

    def _json_supplier_switches(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON supplier switches."""
        return [
            {
                "from_supplier": sw.get("from_supplier", ""),
                "to_supplier": sw.get("to_supplier", ""),
                "from_country": sw.get("from_country", ""),
                "to_country": sw.get("to_country", ""),
                "volume_tonnes": round(sw.get("volume_tonnes", 0.0), 2),
                "emission_delta_tco2e": round(sw.get("emission_delta_tco2e", 0.0), 2),
                "cost_delta_eur": round(sw.get("cost_delta_eur", 0.0), 2),
                "logistics_delta_eur": round(sw.get("logistics_delta_eur", 0.0), 2),
                "net_impact_eur": round(sw.get("net_impact_eur", 0.0), 2),
            }
            for sw in data.get("supplier_switches", [])
        ]

    def _json_decarbonization_roi(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON decarbonization ROI."""
        return [
            {
                "supplier_name": r.get("supplier_name", ""),
                "current_intensity_tco2e_per_t": round(
                    r.get("current_intensity_tco2e_per_t", 0.0), 4
                ),
                "target_intensity_tco2e_per_t": round(
                    r.get("target_intensity_tco2e_per_t", 0.0), 4
                ),
                "reduction_pct": round(r.get("reduction_pct", 0.0), 2),
                "annual_cbam_savings_eur": round(r.get("annual_cbam_savings_eur", 0.0), 2),
                "estimated_investment_eur": round(r.get("estimated_investment_eur", 0.0), 2),
                "payback_years": round(r.get("payback_years", 0.0), 2),
                "irr_pct": round(r["irr_pct"], 2) if r.get("irr_pct") is not None else None,
            }
            for r in data.get("decarbonization_roi", [])
        ]

    def _json_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON recommendations."""
        return [
            {
                "rank": r.get("rank", 0),
                "action": r.get("action", ""),
                "expected_savings_eur": round(r.get("expected_savings_eur", 0.0), 2),
                "implementation_complexity": r.get("implementation_complexity", "medium"),
                "timeline_months": r.get("timeline_months", 0),
                "confidence": r.get("confidence", "medium"),
            }
            for r in sorted(
                data.get("recommendations", []),
                key=lambda x: x.get("rank", 0),
            )
        ]

    def _json_assumptions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON assumptions."""
        return [
            {
                "assumption_id": a.get("assumption_id", ""),
                "category": a.get("category", ""),
                "description": a.get("description", ""),
                "value": a.get("value", ""),
                "source": a.get("source", ""),
                "sensitivity": a.get("sensitivity", "low"),
            }
            for a in data.get("assumptions", [])
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _compute_provenance_hash(self, content: str) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _currency(self) -> str:
        """Get configured currency code."""
        return self.config.get("currency", "EUR")

    def _fmt_int(self, value: Union[int, float, None]) -> str:
        """Format integer with thousand separators."""
        if value is None:
            return "0"
        return f"{int(value):,}"

    def _fmt_num(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format number with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _fmt_cur(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format currency value."""
        return f"{currency} {value:,.2f}"

    def _fmt_date(self, dt: Union[datetime, str]) -> str:
        """Format datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if dt else ""
        return dt.strftime("%Y-%m-%d")

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = self._get_css()
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f'Pack: {self.PACK_ID} | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )

    def _get_css(self) -> str:
        """Return inline CSS for HTML reports."""
        return (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".kpi-unit{font-size:12px;color:#95a5a6;margin-top:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px;color:#2c3e50}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
