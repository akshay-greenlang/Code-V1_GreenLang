# -*- coding: utf-8 -*-
"""
UncertaintyAnalysisReportTemplate - Uncertainty Analysis Report for PACK-041.

Generates a comprehensive uncertainty analysis report covering methodology
description (analytical propagation and Monte Carlo simulation), per-source
uncertainty inputs, analytical results with combined uncertainty and confidence
intervals, Monte Carlo results with distribution histogram data and percentiles,
top uncertainty contributors ranked by impact, data quality improvement
recommendations, and sensitivity analysis.

Sections:
    1. Methodology Description
    2. Per-Source Uncertainty Inputs
    3. Analytical Propagation Results
    4. Monte Carlo Simulation Results
    5. Top Uncertainty Contributors
    6. Sensitivity Analysis
    7. Data Quality Improvement Recommendations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with histogram/percentile data)

Regulatory References:
    - IPCC 2006 Guidelines Vol. 1, Ch. 3 (Uncertainties)
    - GHG Protocol Corporate Standard, Ch. 7
    - ISO 14064-1:2018 Clause 5.4 (Uncertainty)

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "41.0.0"


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _fmt_pct(value: Optional[float], signed: bool = True) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    if signed:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"
    return f"{value:.1f}%"


class UncertaintyAnalysisReportTemplate:
    """
    Uncertainty analysis report template.

    Renders comprehensive uncertainty analysis reports covering analytical
    propagation, Monte Carlo simulation, per-source uncertainty inputs,
    top contributors, sensitivity analysis, and improvement recommendations.
    All outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = UncertaintyAnalysisReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UncertaintyAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

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
        """Render uncertainty analysis report as Markdown.

        Args:
            data: Validated uncertainty analysis data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_per_source_inputs(data),
            self._md_analytical_results(data),
            self._md_monte_carlo_results(data),
            self._md_top_contributors(data),
            self._md_sensitivity_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render uncertainty analysis report as HTML.

        Args:
            data: Validated uncertainty analysis data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_methodology(data),
            self._html_per_source_inputs(data),
            self._html_analytical_results(data),
            self._html_monte_carlo_results(data),
            self._html_top_contributors(data),
            self._html_sensitivity_analysis(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render uncertainty analysis report as JSON-serializable dict.

        Args:
            data: Validated uncertainty analysis data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "uncertainty_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "methodology": data.get("methodology", {}),
            "per_source_inputs": data.get("per_source_inputs", []),
            "analytical_results": data.get("analytical_results", {}),
            "monte_carlo_results": data.get("monte_carlo_results", {}),
            "top_contributors": data.get("top_contributors", []),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "recommendations": data.get("recommendations", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Uncertainty Analysis Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology description."""
        meth = data.get("methodology", {})
        analytical = meth.get("analytical_description",
                              "Error propagation using IPCC Approach 1 (quadrature addition).")
        monte_carlo = meth.get("monte_carlo_description",
                               "Monte Carlo simulation with 10,000 iterations.")
        confidence = meth.get("confidence_level", "95%")
        iterations = meth.get("monte_carlo_iterations", 10000)
        distribution = meth.get("default_distribution", "Normal")
        lines = [
            "## 1. Methodology Description",
            "",
            "### 1.1 Analytical Propagation",
            "",
            analytical,
            "",
            "### 1.2 Monte Carlo Simulation",
            "",
            monte_carlo,
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Confidence Level | {confidence} |",
            f"| Monte Carlo Iterations | {iterations:,} |",
            f"| Default Distribution | {distribution} |",
            f"| Correlation Handling | {meth.get('correlation_handling', 'Independent')} |",
        ]
        return "\n".join(lines)

    def _md_per_source_inputs(self, data: Dict[str, Any]) -> str:
        """Render Markdown per-source uncertainty inputs."""
        inputs = data.get("per_source_inputs", [])
        if not inputs:
            return "## 2. Per-Source Uncertainty Inputs\n\nNo per-source inputs provided."
        lines = [
            "## 2. Per-Source Uncertainty Inputs",
            "",
            "| Source | Emissions tCO2e | AD Uncertainty % | EF Uncertainty % | Combined % | Distribution | Notes |",
            "|--------|----------------|-----------------|-----------------|-----------|-------------|-------|",
        ]
        for inp in sorted(inputs, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            source = inp.get("source_name", "")
            emissions = _fmt_tco2e(inp.get("emissions_tco2e"))
            ad_unc = f"+/-{inp.get('ad_uncertainty_pct', 0):.1f}%"
            ef_unc = f"+/-{inp.get('ef_uncertainty_pct', 0):.1f}%"
            combined = f"+/-{inp.get('combined_uncertainty_pct', 0):.1f}%"
            dist = inp.get("distribution", "Normal")
            notes = inp.get("notes", "-")
            lines.append(f"| {source} | {emissions} | {ad_unc} | {ef_unc} | {combined} | {dist} | {notes} |")
        return "\n".join(lines)

    def _md_analytical_results(self, data: Dict[str, Any]) -> str:
        """Render Markdown analytical propagation results."""
        results = data.get("analytical_results", {})
        if not results:
            return "## 3. Analytical Propagation Results\n\nNo analytical results available."
        total = results.get("total_emissions_tco2e", 0.0)
        combined = results.get("combined_uncertainty_pct", 0.0)
        confidence = results.get("confidence_level", "95%")
        lower = results.get("lower_bound_tco2e", 0.0)
        upper = results.get("upper_bound_tco2e", 0.0)
        lines = [
            "## 3. Analytical Propagation Results",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Total Emissions | {_fmt_tco2e(total)} |",
            f"| Combined Uncertainty | +/-{combined:.1f}% |",
            f"| Confidence Level | {confidence} |",
            f"| Lower Bound | {_fmt_tco2e(lower)} |",
            f"| Upper Bound | {_fmt_tco2e(upper)} |",
            f"| Uncertainty Range | {_fmt_tco2e(upper - lower)} |",
        ]
        # Per-scope breakdown
        scope_breakdown = results.get("scope_breakdown", [])
        if scope_breakdown:
            lines.extend([
                "",
                "### By Scope",
                "",
                "| Scope | Emissions | Uncertainty % | Lower | Upper |",
                "|-------|----------|--------------|-------|-------|",
            ])
            for sb in scope_breakdown:
                scope = sb.get("scope", "")
                em = _fmt_tco2e(sb.get("emissions_tco2e"))
                unc = f"+/-{sb.get('uncertainty_pct', 0):.1f}%"
                lb = _fmt_tco2e(sb.get("lower_bound_tco2e"))
                ub = _fmt_tco2e(sb.get("upper_bound_tco2e"))
                lines.append(f"| {scope} | {em} | {unc} | {lb} | {ub} |")
        return "\n".join(lines)

    def _md_monte_carlo_results(self, data: Dict[str, Any]) -> str:
        """Render Markdown Monte Carlo simulation results."""
        mc = data.get("monte_carlo_results", {})
        if not mc:
            return "## 4. Monte Carlo Simulation Results\n\nNo Monte Carlo results available."
        iterations = mc.get("iterations", 0)
        mean = mc.get("mean_tco2e", 0.0)
        std_dev = mc.get("std_dev_tco2e", 0.0)
        median = mc.get("median_tco2e", 0.0)
        percentiles = mc.get("percentiles", {})
        lines = [
            "## 4. Monte Carlo Simulation Results",
            "",
            f"**Iterations:** {iterations:,}",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Mean | {_fmt_tco2e(mean)} |",
            f"| Standard Deviation | {_fmt_tco2e(std_dev)} |",
            f"| Median (P50) | {_fmt_tco2e(median)} |",
            f"| CV (Std/Mean) | {(std_dev / mean * 100):.1f}% |" if mean > 0 else "| CV | N/A |",
        ]
        # Percentiles
        if percentiles:
            lines.extend([
                "",
                "### Percentile Distribution",
                "",
                "| Percentile | Value (tCO2e) |",
                "|-----------|--------------|",
            ])
            for pct_key in sorted(percentiles.keys(), key=lambda x: float(x.replace("P", ""))):
                val = percentiles[pct_key]
                lines.append(f"| {pct_key} | {_fmt_tco2e(val)} |")
        # Histogram bins
        histogram = mc.get("histogram_bins", [])
        if histogram:
            lines.extend([
                "",
                "### Distribution Histogram Data",
                "",
                "| Bin Lower | Bin Upper | Count | Frequency % |",
                "|----------|---------|-------|------------|",
            ])
            total_count = sum(b.get("count", 0) for b in histogram)
            for b in histogram:
                lower = _fmt_tco2e(b.get("lower"))
                upper = _fmt_tco2e(b.get("upper"))
                count = b.get("count", 0)
                freq = f"{(count / total_count * 100):.1f}%" if total_count > 0 else "0.0%"
                lines.append(f"| {lower} | {upper} | {count:,} | {freq} |")
        return "\n".join(lines)

    def _md_top_contributors(self, data: Dict[str, Any]) -> str:
        """Render Markdown top uncertainty contributors."""
        contributors = data.get("top_contributors", [])
        if not contributors:
            return "## 5. Top Uncertainty Contributors\n\nNo contributor ranking available."
        lines = [
            "## 5. Top Uncertainty Contributors",
            "",
            "| Rank | Source | Emissions tCO2e | Uncertainty % | Contribution to Total % | Category |",
            "|------|--------|----------------|--------------|------------------------|----------|",
        ]
        for i, c in enumerate(contributors, 1):
            source = c.get("source_name", "")
            emissions = _fmt_tco2e(c.get("emissions_tco2e"))
            unc = f"+/-{c.get('uncertainty_pct', 0):.1f}%"
            contrib = f"{c.get('contribution_to_total_pct', 0):.1f}%"
            cat = c.get("category", "-")
            lines.append(f"| {i} | {source} | {emissions} | {unc} | {contrib} | {cat} |")
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown sensitivity analysis."""
        sensitivity = data.get("sensitivity_analysis", [])
        if not sensitivity:
            return "## 6. Sensitivity Analysis\n\nNo sensitivity analysis performed."
        lines = [
            "## 6. Sensitivity Analysis",
            "",
            "| Parameter | Base Value | -10% Impact | +10% Impact | Sensitivity Coefficient | Rank |",
            "|-----------|-----------|------------|------------|----------------------|------|",
        ]
        for i, s in enumerate(sorted(sensitivity, key=lambda x: abs(x.get("sensitivity_coefficient", 0)), reverse=True), 1):
            param = s.get("parameter_name", "")
            base = _fmt_num(s.get("base_value"))
            minus = _fmt_tco2e(s.get("minus_10pct_impact_tco2e"))
            plus = _fmt_tco2e(s.get("plus_10pct_impact_tco2e"))
            coeff = f"{s.get('sensitivity_coefficient', 0):.3f}"
            lines.append(f"| {param} | {base} | {minus} | {plus} | {coeff} | {i} |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 7. Data Quality Improvement Recommendations\n\nNo recommendations."
        lines = [
            "## 7. Data Quality Improvement Recommendations",
            "",
            "| Priority | Source | Current Uncertainty | Target Uncertainty | Action | Estimated Effort | Expected Reduction |",
            "|----------|--------|-------------------|-------------------|--------|-----------------|-------------------|",
        ]
        for rec in recs:
            priority = rec.get("priority", "-")
            source = rec.get("source_name", "")
            current = f"+/-{rec.get('current_uncertainty_pct', 0):.1f}%"
            target = f"+/-{rec.get('target_uncertainty_pct', 0):.1f}%"
            action = rec.get("action", "")
            effort = rec.get("estimated_effort", "-")
            reduction = f"{rec.get('expected_reduction_pct', 0):.1f}%"
            lines.append(f"| {priority} | {source} | {current} | {target} | {action} | {effort} | {reduction} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Uncertainty Analysis - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #f4a261;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#415a77;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".high-unc{color:#e63946;font-weight:600;}\n"
            ".med-unc{color:#f4a261;font-weight:600;}\n"
            ".low-unc{color:#2a9d8f;font-weight:600;}\n"
            ".bar{display:inline-block;height:16px;background:#457b9d;border-radius:2px;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".stat-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:160px;}\n"
            ".stat-value{font-size:1.4rem;font-weight:700;color:#1b263b;}\n"
            ".stat-label{font-size:0.8rem;color:#555;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Uncertainty Analysis Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology description."""
        meth = data.get("methodology", {})
        analytical = meth.get("analytical_description",
                              "Error propagation using IPCC Approach 1.")
        monte_carlo = meth.get("monte_carlo_description",
                               "Monte Carlo simulation with 10,000 iterations.")
        return (
            '<div class="section">\n'
            "<h2>1. Methodology</h2>\n"
            f"<h3>Analytical Propagation</h3><p>{analytical}</p>\n"
            f"<h3>Monte Carlo Simulation</h3><p>{monte_carlo}</p>\n"
            "</div>"
        )

    def _html_per_source_inputs(self, data: Dict[str, Any]) -> str:
        """Render HTML per-source uncertainty inputs."""
        inputs = data.get("per_source_inputs", [])
        if not inputs:
            return ""
        rows = ""
        for inp in sorted(inputs, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            source = inp.get("source_name", "")
            emissions = _fmt_tco2e(inp.get("emissions_tco2e"))
            ad_unc = f"+/-{inp.get('ad_uncertainty_pct', 0):.1f}%"
            ef_unc = f"+/-{inp.get('ef_uncertainty_pct', 0):.1f}%"
            combined = inp.get("combined_uncertainty_pct", 0)
            unc_class = "high-unc" if combined > 30 else ("med-unc" if combined > 15 else "low-unc")
            rows += (
                f'<tr><td>{source}</td><td>{emissions}</td><td>{ad_unc}</td>'
                f'<td>{ef_unc}</td><td class="{unc_class}">+/-{combined:.1f}%</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>2. Per-Source Uncertainty Inputs</h2>\n"
            "<table><thead><tr><th>Source</th><th>Emissions</th><th>AD Unc.</th>"
            "<th>EF Unc.</th><th>Combined</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_analytical_results(self, data: Dict[str, Any]) -> str:
        """Render HTML analytical results."""
        results = data.get("analytical_results", {})
        if not results:
            return ""
        total = results.get("total_emissions_tco2e", 0.0)
        combined = results.get("combined_uncertainty_pct", 0.0)
        lower = results.get("lower_bound_tco2e", 0.0)
        upper = results.get("upper_bound_tco2e", 0.0)
        confidence = results.get("confidence_level", "95%")
        return (
            '<div class="section">\n'
            "<h2>3. Analytical Propagation Results</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Total Emissions</td><td>{_fmt_tco2e(total)}</td></tr>\n"
            f"<tr><td>Combined Uncertainty</td><td>+/-{combined:.1f}%</td></tr>\n"
            f"<tr><td>Confidence Level</td><td>{confidence}</td></tr>\n"
            f"<tr><td>Lower Bound</td><td>{_fmt_tco2e(lower)}</td></tr>\n"
            f"<tr><td>Upper Bound</td><td>{_fmt_tco2e(upper)}</td></tr>\n"
            f"<tr><td>Range</td><td>{_fmt_tco2e(upper - lower)}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_monte_carlo_results(self, data: Dict[str, Any]) -> str:
        """Render HTML Monte Carlo results."""
        mc = data.get("monte_carlo_results", {})
        if not mc:
            return ""
        mean = mc.get("mean_tco2e", 0.0)
        std_dev = mc.get("std_dev_tco2e", 0.0)
        median = mc.get("median_tco2e", 0.0)
        percentiles = mc.get("percentiles", {})
        stats_rows = (
            f"<tr><td>Mean</td><td>{_fmt_tco2e(mean)}</td></tr>\n"
            f"<tr><td>Std Dev</td><td>{_fmt_tco2e(std_dev)}</td></tr>\n"
            f"<tr><td>Median</td><td>{_fmt_tco2e(median)}</td></tr>\n"
        )
        pct_rows = ""
        for pct_key in sorted(percentiles.keys(), key=lambda x: float(x.replace("P", ""))):
            val = percentiles[pct_key]
            pct_rows += f"<tr><td>{pct_key}</td><td>{_fmt_tco2e(val)}</td></tr>\n"
        # Histogram visualization (inline bar chart)
        histogram = mc.get("histogram_bins", [])
        hist_rows = ""
        if histogram:
            max_count = max((b.get("count", 0) for b in histogram), default=1)
            total_count = sum(b.get("count", 0) for b in histogram)
            for b in histogram:
                count = b.get("count", 0)
                freq = (count / total_count * 100) if total_count > 0 else 0
                bar_width = int((count / max_count) * 200) if max_count > 0 else 0
                hist_rows += (
                    f"<tr><td>{_fmt_tco2e(b.get('lower'))}</td><td>{_fmt_tco2e(b.get('upper'))}</td>"
                    f'<td>{count:,}</td><td>{freq:.1f}%</td>'
                    f'<td><span class="bar" style="width:{bar_width}px"></span></td></tr>\n'
                )
        parts = [
            '<div class="section">',
            "<h2>4. Monte Carlo Results</h2>",
            f"<p><strong>Iterations:</strong> {mc.get('iterations', 0):,}</p>",
            "<h3>Statistics</h3>",
            "<table><thead><tr><th>Statistic</th><th>Value</th></tr></thead>",
            f"<tbody>{stats_rows}</tbody></table>",
        ]
        if pct_rows:
            parts.extend([
                "<h3>Percentiles</h3>",
                "<table><thead><tr><th>Percentile</th><th>Value</th></tr></thead>",
                f"<tbody>{pct_rows}</tbody></table>",
            ])
        if hist_rows:
            parts.extend([
                "<h3>Distribution</h3>",
                "<table><thead><tr><th>Lower</th><th>Upper</th><th>Count</th><th>Freq</th><th>Distribution</th></tr></thead>",
                f"<tbody>{hist_rows}</tbody></table>",
            ])
        parts.append("</div>")
        return "\n".join(parts)

    def _html_top_contributors(self, data: Dict[str, Any]) -> str:
        """Render HTML top uncertainty contributors."""
        contributors = data.get("top_contributors", [])
        if not contributors:
            return ""
        rows = ""
        for i, c in enumerate(contributors, 1):
            source = c.get("source_name", "")
            emissions = _fmt_tco2e(c.get("emissions_tco2e"))
            unc = f"+/-{c.get('uncertainty_pct', 0):.1f}%"
            contrib = c.get("contribution_to_total_pct", 0)
            bar_width = int(contrib * 2)
            rows += (
                f"<tr><td>{i}</td><td>{source}</td><td>{emissions}</td>"
                f"<td>{unc}</td><td>{contrib:.1f}% "
                f'<span class="bar" style="width:{bar_width}px"></span></td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>5. Top Uncertainty Contributors</h2>\n"
            "<table><thead><tr><th>#</th><th>Source</th><th>Emissions</th>"
            "<th>Uncertainty</th><th>Contribution</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis."""
        sensitivity = data.get("sensitivity_analysis", [])
        if not sensitivity:
            return ""
        rows = ""
        for s in sorted(sensitivity, key=lambda x: abs(x.get("sensitivity_coefficient", 0)), reverse=True):
            param = s.get("parameter_name", "")
            base = _fmt_num(s.get("base_value"))
            minus = _fmt_tco2e(s.get("minus_10pct_impact_tco2e"))
            plus = _fmt_tco2e(s.get("plus_10pct_impact_tco2e"))
            coeff = f"{s.get('sensitivity_coefficient', 0):.3f}"
            rows += (
                f"<tr><td>{param}</td><td>{base}</td><td>{minus}</td>"
                f"<td>{plus}</td><td>{coeff}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Sensitivity Analysis</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Base</th>"
            "<th>-10% Impact</th><th>+10% Impact</th><th>Coefficient</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        rows = ""
        for rec in recs:
            priority = rec.get("priority", "-")
            source = rec.get("source_name", "")
            current = f"+/-{rec.get('current_uncertainty_pct', 0):.1f}%"
            target = f"+/-{rec.get('target_uncertainty_pct', 0):.1f}%"
            action = rec.get("action", "")
            rows += (
                f"<tr><td>{priority}</td><td>{source}</td><td>{current}</td>"
                f"<td>{target}</td><td>{action}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Recommendations</h2>\n"
            "<table><thead><tr><th>Priority</th><th>Source</th>"
            "<th>Current</th><th>Target</th><th>Action</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
