# -*- coding: utf-8 -*-
"""
ScenarioAnalysisReport - Scenario & Monte Carlo Analysis for PACK-046.

Generates a scenario analysis report with scenario definitions, base
case, scenario results comparison table, Monte Carlo distribution data,
probability of target achievement, sensitivity tornado chart data, key
drivers, and recommendations.

Sections:
    1. Scenario Definitions
    2. Base Case
    3. Scenario Results Table
    4. Monte Carlo Distribution Data
    5. Probability of Target Achievement
    6. Sensitivity Tornado Data
    7. Key Drivers
    8. Recommendations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with fan/tornado chart data)

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


class ScenarioType(str, Enum):
    """Scenario classification types."""
    BASE_CASE = "base_case"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    BEST_CASE = "best_case"
    WORST_CASE = "worst_case"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class ScenarioDefinition(BaseModel):
    """Definition of a single scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    scenario_type: ScenarioType = Field(ScenarioType.CUSTOM, description="Scenario type")
    description: str = Field("", description="Scenario description")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Scenario parameter overrides"
    )


class ScenarioResult(BaseModel):
    """Results for a single scenario."""
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field("", description="Scenario name")
    projected_intensity: float = Field(0.0, description="Projected intensity at target year")
    projected_emissions_tco2e: float = Field(0.0, description="Projected total emissions")
    reduction_from_base_pct: float = Field(0.0, description="% reduction from base year")
    meets_target: bool = Field(False, description="Whether scenario meets target")
    target_year: int = Field(0, description="Target year for projection")
    intensity_unit: str = Field("", description="Intensity unit")


class MonteCarloDistribution(BaseModel):
    """Monte Carlo simulation distribution summary."""
    simulation_runs: int = Field(10000, description="Number of simulation runs")
    mean_intensity: float = Field(0.0, description="Mean projected intensity")
    median_intensity: float = Field(0.0, description="Median projected intensity")
    std_dev: float = Field(0.0, description="Standard deviation")
    p5: float = Field(0.0, description="5th percentile")
    p10: float = Field(0.0, description="10th percentile")
    p25: float = Field(0.0, description="25th percentile")
    p50: float = Field(0.0, description="50th percentile (median)")
    p75: float = Field(0.0, description="75th percentile")
    p90: float = Field(0.0, description="90th percentile")
    p95: float = Field(0.0, description="95th percentile")
    histogram_bins: List[float] = Field(default_factory=list, description="Histogram bin edges")
    histogram_counts: List[int] = Field(default_factory=list, description="Histogram bin counts")


class SensitivityItem(BaseModel):
    """Single variable in sensitivity (tornado) analysis."""
    variable_name: str = Field(..., description="Input variable name")
    base_value: float = Field(0.0, description="Base case value")
    low_value: float = Field(0.0, description="Low scenario value")
    high_value: float = Field(0.0, description="High scenario value")
    impact_low: float = Field(0.0, description="Intensity at low value")
    impact_high: float = Field(0.0, description="Intensity at high value")
    swing: float = Field(0.0, description="Absolute swing (high - low impact)")


class KeyDriver(BaseModel):
    """Key driver identified from scenario analysis."""
    driver_name: str = Field(..., description="Driver name")
    impact_description: str = Field("", description="Impact description")
    controllability: str = Field("", description="High / Medium / Low controllability")
    priority: int = Field(1, ge=1, le=5, description="Priority ranking")


class ScenarioRecommendation(BaseModel):
    """Recommendation from scenario analysis."""
    priority: int = Field(1, ge=1, le=5, description="Priority")
    recommendation: str = Field(..., description="Recommendation text")
    scenario_basis: str = Field("", description="Which scenario(s) inform this recommendation")
    expected_benefit: str = Field("", description="Expected benefit description")


class ScenarioReportInput(BaseModel):
    """Complete input model for ScenarioAnalysisReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    scenario_definitions: List[ScenarioDefinition] = Field(
        default_factory=list, description="Scenario definitions"
    )
    base_case: Optional[ScenarioResult] = Field(None, description="Base case results")
    scenario_results: List[ScenarioResult] = Field(
        default_factory=list, description="Scenario results"
    )
    monte_carlo: Optional[MonteCarloDistribution] = Field(
        None, description="Monte Carlo distribution"
    )
    probability_of_target: Optional[float] = Field(
        None, description="Probability of achieving target (%)"
    )
    sensitivity_items: List[SensitivityItem] = Field(
        default_factory=list, description="Sensitivity tornado items"
    )
    key_drivers: List[KeyDriver] = Field(
        default_factory=list, description="Key drivers"
    )
    recommendations: List[ScenarioRecommendation] = Field(
        default_factory=list, description="Recommendations"
    )


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ScenarioAnalysisReport:
    """
    Scenario analysis report template.

    Renders scenario definitions, results comparison, Monte Carlo
    distribution, probability assessments, sensitivity tornado data,
    key drivers, and recommendations. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = ScenarioAnalysisReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ScenarioAnalysisReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render scenario analysis as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render scenario analysis as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render scenario analysis as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_definitions(data),
            self._md_base_case(data),
            self._md_results_table(data),
            self._md_monte_carlo(data),
            self._md_probability(data),
            self._md_sensitivity(data),
            self._md_key_drivers(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Scenario Analysis Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_definitions(self, data: Dict[str, Any]) -> str:
        """Render Markdown scenario definitions."""
        defs = data.get("scenario_definitions", [])
        if not defs:
            return "## 1. Scenario Definitions\n\nNo scenarios defined."
        lines = ["## 1. Scenario Definitions", ""]
        for sd in defs:
            name = sd.get("scenario_name", "")
            stype = sd.get("scenario_type", "custom")
            desc = sd.get("description", "")
            assumptions = sd.get("assumptions", [])
            lines.append(f"### {name} ({stype})")
            if desc:
                lines.append(f"\n{desc}")
            if assumptions:
                lines.append("\n**Assumptions:**")
                for a in assumptions:
                    lines.append(f"- {a}")
            lines.append("")
        return "\n".join(lines)

    def _md_base_case(self, data: Dict[str, Any]) -> str:
        """Render Markdown base case summary."""
        bc = data.get("base_case")
        if not bc:
            return "## 2. Base Case\n\nNo base case defined."
        lines = [
            "## 2. Base Case",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Projected Intensity | {bc.get('projected_intensity', 0):,.4f} {bc.get('intensity_unit', '')} |",
            f"| Projected Emissions | {bc.get('projected_emissions_tco2e', 0):,.1f} tCO2e |",
            f"| Reduction from Base | {bc.get('reduction_from_base_pct', 0):.1f}% |",
            f"| Meets Target | {'Yes' if bc.get('meets_target') else 'No'} |",
            f"| Target Year | {bc.get('target_year', '')} |",
        ]
        return "\n".join(lines)

    def _md_results_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown scenario results comparison table."""
        results = data.get("scenario_results", [])
        if not results:
            return "## 3. Scenario Results\n\nNo scenario results available."
        lines = [
            "## 3. Scenario Results",
            "",
            "| Scenario | Intensity | Emissions (tCO2e) | Reduction % | Meets Target |",
            "|----------|-----------|-------------------|-------------|--------------|",
        ]
        for r in results:
            name = r.get("scenario_name", r.get("scenario_id", ""))
            intensity = r.get("projected_intensity", 0)
            emissions = r.get("projected_emissions_tco2e", 0)
            reduction = r.get("reduction_from_base_pct", 0)
            meets = "Yes" if r.get("meets_target") else "No"
            lines.append(
                f"| {name} | {intensity:,.4f} | {emissions:,.1f} | "
                f"{reduction:.1f}% | **{meets}** |"
            )
        return "\n".join(lines)

    def _md_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render Markdown Monte Carlo distribution summary."""
        mc = data.get("monte_carlo")
        if not mc:
            return ""
        lines = [
            "## 4. Monte Carlo Distribution",
            "",
            f"**Simulation Runs:** {mc.get('simulation_runs', 0):,}",
            "",
            "| Percentile | Intensity |",
            "|------------|-----------|",
            f"| P5 | {mc.get('p5', 0):,.4f} |",
            f"| P10 | {mc.get('p10', 0):,.4f} |",
            f"| P25 | {mc.get('p25', 0):,.4f} |",
            f"| P50 (Median) | {mc.get('p50', 0):,.4f} |",
            f"| P75 | {mc.get('p75', 0):,.4f} |",
            f"| P90 | {mc.get('p90', 0):,.4f} |",
            f"| P95 | {mc.get('p95', 0):,.4f} |",
            "",
            f"**Mean:** {mc.get('mean_intensity', 0):,.4f} | "
            f"**Std Dev:** {mc.get('std_dev', 0):,.4f}",
        ]
        return "\n".join(lines)

    def _md_probability(self, data: Dict[str, Any]) -> str:
        """Render Markdown probability of target achievement."""
        prob = data.get("probability_of_target")
        if prob is None:
            return ""
        return (
            "## 5. Probability of Target Achievement\n\n"
            f"Based on Monte Carlo simulation, the probability of achieving "
            f"the intensity target is **{prob:.1f}%**."
        )

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render Markdown sensitivity tornado table."""
        items = data.get("sensitivity_items", [])
        if not items:
            return ""
        sorted_items = sorted(items, key=lambda x: x.get("swing", 0), reverse=True)
        lines = [
            "## 6. Sensitivity Analysis (Tornado)",
            "",
            "| Variable | Low Impact | High Impact | Swing | Base Value |",
            "|----------|-----------|-------------|-------|------------|",
        ]
        for s in sorted_items:
            name = s.get("variable_name", "")
            low_imp = s.get("impact_low", 0)
            high_imp = s.get("impact_high", 0)
            swing = s.get("swing", 0)
            base = s.get("base_value", 0)
            lines.append(
                f"| {name} | {low_imp:,.4f} | {high_imp:,.4f} | "
                f"{swing:,.4f} | {base:,.4f} |"
            )
        return "\n".join(lines)

    def _md_key_drivers(self, data: Dict[str, Any]) -> str:
        """Render Markdown key drivers."""
        drivers = data.get("key_drivers", [])
        if not drivers:
            return ""
        lines = ["## 7. Key Drivers", ""]
        for d in drivers:
            name = d.get("driver_name", "")
            impact = d.get("impact_description", "")
            control = d.get("controllability", "")
            lines.append(f"- **{name}:** {impact} (Controllability: {control})")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 8. Recommendations", ""]
        for r in recs:
            priority = r.get("priority", 1)
            text = r.get("recommendation", "")
            basis = r.get("scenario_basis", "")
            benefit = r.get("expected_benefit", "")
            lines.append(f"**P{priority}.** {text}")
            details = []
            if basis:
                details.append(f"Basis: {basis}")
            if benefit:
                details.append(f"Benefit: {benefit}")
            if details:
                lines.append(f"   - {' | '.join(details)}")
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_definitions(data),
            self._html_base_case(data),
            self._html_results_table(data),
            self._html_monte_carlo(data),
            self._html_probability(data),
            self._html_sensitivity(data),
            self._html_key_drivers(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scenario Analysis - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".meets-yes{color:#2a9d8f;font-weight:700;}\n"
            ".meets-no{color:#e76f51;font-weight:700;}\n"
            ".prob-box{background:#f0f4f8;border-radius:8px;padding:1.5rem;"
            "text-align:center;margin:1rem 0;}\n"
            ".prob-value{font-size:2rem;font-weight:700;color:#1b263b;}\n"
            ".tornado-bar{display:inline-block;height:20px;border-radius:3px;"
            "vertical-align:middle;}\n"
            ".tornado-low{background:#2a9d8f;}\n"
            ".tornado-high{background:#e76f51;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Scenario Analysis Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period}</p>\n<hr>\n</div>"
        )

    def _html_definitions(self, data: Dict[str, Any]) -> str:
        """Render HTML scenario definitions."""
        defs = data.get("scenario_definitions", [])
        if not defs:
            return ""
        content = ""
        for sd in defs:
            name = sd.get("scenario_name", "")
            desc = sd.get("description", "")
            assumptions = sd.get("assumptions", [])
            content += f"<h3>{name}</h3>\n"
            if desc:
                content += f"<p>{desc}</p>\n"
            if assumptions:
                items = "".join(f"<li>{a}</li>\n" for a in assumptions)
                content += f"<ul>{items}</ul>\n"
        return f'<div class="section">\n<h2>1. Scenario Definitions</h2>\n{content}</div>'

    def _html_base_case(self, data: Dict[str, Any]) -> str:
        """Render HTML base case."""
        bc = data.get("base_case")
        if not bc:
            return ""
        meets = bc.get("meets_target", False)
        meets_css = "meets-yes" if meets else "meets-no"
        meets_str = "Yes" if meets else "No"
        rows = (
            f"<tr><td>Projected Intensity</td>"
            f"<td>{bc.get('projected_intensity', 0):,.4f} {bc.get('intensity_unit', '')}</td></tr>\n"
            f"<tr><td>Projected Emissions</td>"
            f"<td>{bc.get('projected_emissions_tco2e', 0):,.1f} tCO2e</td></tr>\n"
            f"<tr><td>Reduction from Base</td>"
            f"<td>{bc.get('reduction_from_base_pct', 0):.1f}%</td></tr>\n"
            f'<tr><td>Meets Target</td><td class="{meets_css}">'
            f"<strong>{meets_str}</strong></td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>2. Base Case</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_results_table(self, data: Dict[str, Any]) -> str:
        """Render HTML scenario results comparison table."""
        results = data.get("scenario_results", [])
        if not results:
            return ""
        rows = ""
        for r in results:
            name = r.get("scenario_name", r.get("scenario_id", ""))
            intensity = r.get("projected_intensity", 0)
            emissions = r.get("projected_emissions_tco2e", 0)
            reduction = r.get("reduction_from_base_pct", 0)
            meets = r.get("meets_target", False)
            meets_css = "meets-yes" if meets else "meets-no"
            meets_str = "Yes" if meets else "No"
            rows += (
                f"<tr><td>{name}</td><td>{intensity:,.4f}</td>"
                f"<td>{emissions:,.1f}</td><td>{reduction:.1f}%</td>"
                f'<td class="{meets_css}"><strong>{meets_str}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Scenario Results</h2>\n'
            "<table><thead><tr><th>Scenario</th><th>Intensity</th>"
            "<th>Emissions</th><th>Reduction</th><th>Meets Target</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render HTML Monte Carlo distribution summary."""
        mc = data.get("monte_carlo")
        if not mc:
            return ""
        runs = mc.get("simulation_runs", 0)
        rows = ""
        for pct_label, key in [("P5", "p5"), ("P10", "p10"), ("P25", "p25"),
                                ("P50 (Median)", "p50"), ("P75", "p75"),
                                ("P90", "p90"), ("P95", "p95")]:
            rows += f"<tr><td>{pct_label}</td><td>{mc.get(key, 0):,.4f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Monte Carlo Distribution</h2>\n'
            f"<p><strong>Simulation Runs:</strong> {runs:,}</p>\n"
            "<table><thead><tr><th>Percentile</th><th>Intensity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p>Mean: {mc.get('mean_intensity', 0):,.4f} | "
            f"Std Dev: {mc.get('std_dev', 0):,.4f}</p>\n</div>"
        )

    def _html_probability(self, data: Dict[str, Any]) -> str:
        """Render HTML probability of target achievement."""
        prob = data.get("probability_of_target")
        if prob is None:
            return ""
        return (
            '<div class="section">\n<h2>5. Probability of Target Achievement</h2>\n'
            f'<div class="prob-box">\n'
            f'<div class="prob-value">{prob:.1f}%</div>\n'
            f"<p>Probability of achieving the intensity target</p>\n"
            "</div>\n</div>"
        )

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity tornado chart table."""
        items = data.get("sensitivity_items", [])
        if not items:
            return ""
        sorted_items = sorted(items, key=lambda x: x.get("swing", 0), reverse=True)
        max_swing = max((s.get("swing", 0) for s in sorted_items), default=1) or 1
        rows = ""
        for s in sorted_items:
            name = s.get("variable_name", "")
            low_imp = s.get("impact_low", 0)
            high_imp = s.get("impact_high", 0)
            swing = s.get("swing", 0)
            bar_width = int((swing / max_swing) * 200)
            half = bar_width // 2
            rows += (
                f"<tr><td>{name}</td><td>{low_imp:,.4f}</td><td>{high_imp:,.4f}</td>"
                f'<td><span class="tornado-bar tornado-low" style="width:{half}px;"></span>'
                f'<span class="tornado-bar tornado-high" style="width:{half}px;"></span>'
                f" {swing:,.4f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Sensitivity Analysis</h2>\n'
            "<table><thead><tr><th>Variable</th><th>Low Impact</th>"
            "<th>High Impact</th><th>Swing</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_key_drivers(self, data: Dict[str, Any]) -> str:
        """Render HTML key drivers."""
        drivers = data.get("key_drivers", [])
        if not drivers:
            return ""
        items = ""
        for d in drivers:
            name = d.get("driver_name", "")
            impact = d.get("impact_description", "")
            control = d.get("controllability", "")
            items += f"<li><strong>{name}:</strong> {impact} (Controllability: {control})</li>\n"
        return (
            '<div class="section">\n<h2>7. Key Drivers</h2>\n'
            f"<ul>{items}</ul>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", 1)
            text = r.get("recommendation", "")
            basis = r.get("scenario_basis", "")
            benefit = r.get("expected_benefit", "")
            detail = ""
            if benefit:
                detail = f" &mdash; <em>{benefit}</em>"
            if basis:
                detail += f" (Based on: {basis})"
            items += f"<li><strong>P{priority}:</strong> {text}{detail}</li>\n"
        return (
            '<div class="section">\n<h2>8. Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render scenario analysis as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "scenario_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "scenario_definitions": data.get("scenario_definitions", []),
            "base_case": data.get("base_case"),
            "scenario_results": data.get("scenario_results", []),
            "monte_carlo": data.get("monte_carlo"),
            "probability_of_target": data.get("probability_of_target"),
            "sensitivity_items": data.get("sensitivity_items", []),
            "key_drivers": data.get("key_drivers", []),
            "recommendations": data.get("recommendations", []),
            "chart_data": {
                "fan_chart": self._build_fan_chart(data),
                "tornado_chart": self._build_tornado_chart(data),
                "monte_carlo_histogram": self._build_mc_histogram(data),
            },
        }

    def _build_fan_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build fan chart data from scenario results."""
        results = data.get("scenario_results", [])
        if not results:
            return {}
        return {
            "scenarios": [
                {
                    "name": r.get("scenario_name", r.get("scenario_id", "")),
                    "intensity": r.get("projected_intensity", 0),
                    "emissions": r.get("projected_emissions_tco2e", 0),
                }
                for r in results
            ],
        }

    def _build_tornado_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build tornado chart data from sensitivity items."""
        items = data.get("sensitivity_items", [])
        if not items:
            return {}
        sorted_items = sorted(items, key=lambda x: x.get("swing", 0), reverse=True)
        return {
            "variables": [s.get("variable_name", "") for s in sorted_items],
            "low_impacts": [s.get("impact_low", 0) for s in sorted_items],
            "high_impacts": [s.get("impact_high", 0) for s in sorted_items],
            "swings": [s.get("swing", 0) for s in sorted_items],
        }

    def _build_mc_histogram(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Monte Carlo histogram chart data."""
        mc = data.get("monte_carlo")
        if not mc:
            return {}
        return {
            "bin_edges": mc.get("histogram_bins", []),
            "counts": mc.get("histogram_counts", []),
            "mean": mc.get("mean_intensity", 0),
            "median": mc.get("median_intensity", 0),
            "std_dev": mc.get("std_dev", 0),
        }
