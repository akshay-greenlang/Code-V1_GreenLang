# -*- coding: utf-8 -*-
"""
Scope3UncertaintyReportTemplate - Monte Carlo and Sensitivity Analysis for PACK-042.

Generates an uncertainty analysis report with total Scope 3 95% CI,
per-category uncertainty ranges, probability distribution data (histogram),
sensitivity tornado chart data, methodology tier vs uncertainty correlation,
tier upgrade impact quantification, and correlation matrix.

Sections:
    1. Total Scope 3 with 95% CI
    2. Per-Category Uncertainty Ranges
    3. Probability Distribution (histogram data)
    4. Sensitivity Analysis (tornado chart data)
    5. Tier vs Uncertainty Correlation
    6. Tier Upgrade Impact
    7. Correlation Matrix

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, statistical gray theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 42.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"


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


class Scope3UncertaintyReportTemplate:
    """
    Monte Carlo results and sensitivity analysis report template.

    Renders uncertainty analysis reports with total Scope 3 confidence
    intervals, per-category ranges, probability distribution histogram
    data, sensitivity tornado chart data, tier-uncertainty correlation,
    tier upgrade impact, and correlation matrices. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope3UncertaintyReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3UncertaintyReportTemplate."""
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

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render uncertainty report as Markdown.

        Args:
            data: Validated uncertainty analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_total_ci(data),
            self._md_per_category(data),
            self._md_probability_distribution(data),
            self._md_sensitivity_analysis(data),
            self._md_tier_correlation(data),
            self._md_tier_upgrade_impact(data),
            self._md_correlation_matrix(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render uncertainty report as HTML.

        Args:
            data: Validated uncertainty analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_total_ci(data),
            self._html_per_category(data),
            self._html_probability_distribution(data),
            self._html_sensitivity_analysis(data),
            self._html_tier_correlation(data),
            self._html_tier_upgrade_impact(data),
            self._html_correlation_matrix(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render uncertainty report as JSON-serializable dict.

        Args:
            data: Validated uncertainty analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "scope3_uncertainty_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "total_scope3_ci": data.get("total_scope3_ci", {}),
            "per_category_uncertainty": data.get("per_category_uncertainty", []),
            "probability_distribution": data.get("probability_distribution", {}),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "tier_correlation": data.get("tier_correlation", []),
            "tier_upgrade_impact": data.get("tier_upgrade_impact", []),
            "correlation_matrix": data.get("correlation_matrix", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Uncertainty Analysis - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_total_ci(self, data: Dict[str, Any]) -> str:
        """Render Markdown total Scope 3 with 95% CI."""
        ci = data.get("total_scope3_ci", {})
        if not ci:
            return "## 1. Total Scope 3 with 95% Confidence Interval\n\nNo CI data available."
        central = ci.get("central_tco2e")
        lower = ci.get("lower_bound_tco2e")
        upper = ci.get("upper_bound_tco2e")
        unc_pct = ci.get("uncertainty_pct")
        method = ci.get("method", "Monte Carlo simulation")
        iterations = ci.get("iterations")
        confidence = ci.get("confidence_level", "95%")
        lines = [
            "## 1. Total Scope 3 with 95% Confidence Interval",
            "",
            f"**Method:** {method} | **Confidence Level:** {confidence}",
        ]
        if iterations:
            lines.append(f"**Iterations:** {iterations:,}")
        lines.extend([
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Central Estimate | {_fmt_tco2e(central)} |",
            f"| Lower Bound (2.5th percentile) | {_fmt_tco2e(lower)} |",
            f"| Upper Bound (97.5th percentile) | {_fmt_tco2e(upper)} |",
        ])
        if unc_pct is not None:
            lines.append(f"| Overall Uncertainty | +/-{unc_pct:.1f}% |")
        p5 = ci.get("p5_tco2e")
        p50 = ci.get("p50_tco2e")
        p95 = ci.get("p95_tco2e")
        if p5 is not None:
            lines.append(f"| P5 | {_fmt_tco2e(p5)} |")
        if p50 is not None:
            lines.append(f"| P50 (Median) | {_fmt_tco2e(p50)} |")
        if p95 is not None:
            lines.append(f"| P95 | {_fmt_tco2e(p95)} |")
        return "\n".join(lines)

    def _md_per_category(self, data: Dict[str, Any]) -> str:
        """Render Markdown per-category uncertainty ranges."""
        categories = data.get("per_category_uncertainty", [])
        if not categories:
            return "## 2. Per-Category Uncertainty Ranges\n\nNo per-category data."
        lines = [
            "## 2. Per-Category Uncertainty Ranges",
            "",
            "| Category | Central tCO2e | Lower | Upper | Uncertainty % | Tier |",
            "|----------|-------------|-------|-------|--------------|------|",
        ]
        for cat in sorted(categories, key=lambda x: x.get("uncertainty_pct", 0), reverse=True):
            name = cat.get("category_name", "")
            central = _fmt_tco2e(cat.get("central_tco2e"))
            lower = _fmt_tco2e(cat.get("lower_bound_tco2e"))
            upper = _fmt_tco2e(cat.get("upper_bound_tco2e"))
            unc = cat.get("uncertainty_pct")
            unc_str = f"+/-{unc:.1f}%" if unc is not None else "-"
            tier = cat.get("tier", "-")
            lines.append(
                f"| {name} | {central} | {lower} | {upper} | {unc_str} | {tier} |"
            )
        return "\n".join(lines)

    def _md_probability_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown probability distribution histogram data."""
        dist = data.get("probability_distribution", {})
        if not dist:
            return "## 3. Probability Distribution\n\nNo distribution data available."
        bins = dist.get("bins", [])
        lines = [
            "## 3. Probability Distribution",
            "",
            f"**Distribution Shape:** {dist.get('shape', 'Approximately normal')}",
            f"**Skewness:** {dist.get('skewness', 'N/A')}",
            f"**Kurtosis:** {dist.get('kurtosis', 'N/A')}",
            "",
        ]
        if bins:
            lines.append("| Bin Range (tCO2e) | Frequency | Probability |")
            lines.append("|------------------|-----------|------------|")
            for b in bins:
                range_str = f"{_fmt_tco2e(b.get('lower'))} - {_fmt_tco2e(b.get('upper'))}"
                freq = b.get("frequency", 0)
                prob = b.get("probability")
                prob_str = f"{prob:.4f}" if prob is not None else "-"
                lines.append(f"| {range_str} | {freq} | {prob_str} |")
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown sensitivity analysis (tornado chart data)."""
        sensitivities = data.get("sensitivity_analysis", [])
        if not sensitivities:
            return "## 4. Sensitivity Analysis\n\nNo sensitivity data available."
        lines = [
            "## 4. Sensitivity Analysis (Tornado Chart Data)",
            "",
            "| Parameter | Low Case tCO2e | Base Case tCO2e | High Case tCO2e | Swing tCO2e |",
            "|-----------|---------------|----------------|----------------|------------|",
        ]
        for s in sorted(sensitivities, key=lambda x: abs(x.get("swing_tco2e", 0)), reverse=True):
            param = s.get("parameter", "")
            low = _fmt_tco2e(s.get("low_case_tco2e"))
            base = _fmt_tco2e(s.get("base_case_tco2e"))
            high = _fmt_tco2e(s.get("high_case_tco2e"))
            swing = _fmt_tco2e(s.get("swing_tco2e"))
            lines.append(f"| {param} | {low} | {base} | {high} | {swing} |")
        return "\n".join(lines)

    def _md_tier_correlation(self, data: Dict[str, Any]) -> str:
        """Render Markdown tier vs uncertainty correlation."""
        correlation = data.get("tier_correlation", [])
        if not correlation:
            return "## 5. Tier vs Uncertainty Correlation\n\nNo correlation data."
        lines = [
            "## 5. Methodology Tier vs Uncertainty Correlation",
            "",
            "| Tier | Avg Uncertainty % | Category Count | Typical Range |",
            "|------|------------------|---------------|---------------|",
        ]
        for t in correlation:
            tier = t.get("tier", "")
            avg_unc = t.get("avg_uncertainty_pct")
            avg_str = f"+/-{avg_unc:.1f}%" if avg_unc is not None else "-"
            count = t.get("category_count", 0)
            range_str = t.get("typical_range", "-")
            lines.append(f"| {tier} | {avg_str} | {count} | {range_str} |")
        return "\n".join(lines)

    def _md_tier_upgrade_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown tier upgrade impact."""
        upgrades = data.get("tier_upgrade_impact", [])
        if not upgrades:
            return "## 6. Tier Upgrade Impact\n\nNo upgrade impact data."
        lines = [
            "## 6. Tier Upgrade Impact Quantification",
            "",
            "| Category | Current Tier | Target Tier | Current Unc % | Projected Unc % | Reduction |",
            "|----------|-------------|-------------|-------------|----------------|-----------|",
        ]
        for u in upgrades:
            cat = u.get("category_name", "")
            current = u.get("current_tier", "-")
            target = u.get("target_tier", "-")
            curr_unc = u.get("current_uncertainty_pct")
            curr_str = f"+/-{curr_unc:.1f}%" if curr_unc is not None else "-"
            proj_unc = u.get("projected_uncertainty_pct")
            proj_str = f"+/-{proj_unc:.1f}%" if proj_unc is not None else "-"
            reduction = u.get("reduction_pct")
            red_str = f"{reduction:.0f}%" if reduction is not None else "-"
            lines.append(
                f"| {cat} | {current} | {target} | {curr_str} | {proj_str} | {red_str} |"
            )
        return "\n".join(lines)

    def _md_correlation_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown correlation matrix."""
        matrix = data.get("correlation_matrix", {})
        categories = matrix.get("categories", [])
        values = matrix.get("values", [])
        if not categories or not values:
            return "## 7. Correlation Matrix\n\nNo correlation matrix available."
        header = "| |"
        sep = "|-|"
        for cat in categories:
            short = cat[:12]
            header += f" {short} |"
            sep += "------|"
        lines = [
            "## 7. Correlation Matrix",
            "",
            header,
            sep,
        ]
        for i, row in enumerate(values):
            cat = categories[i] if i < len(categories) else "?"
            line = f"| {cat[:12]} |"
            for val in row:
                if val is not None:
                    line += f" {val:.2f} |"
                else:
                    line += " - |"
            lines.append(line)
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scope 3 Uncertainty Analysis - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#5D6D7E;border-bottom:3px solid #5D6D7E;padding-bottom:0.5rem;}\n"
            "h2{color:#4A5568;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#5D6D7E;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#eaecee;font-weight:600;color:#4A5568;}\n"
            "tr:nth-child(even){background:#f5f6f7;}\n"
            ".metric-card{display:inline-block;background:#eaecee;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-top:3px solid #5D6D7E;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#4A5568;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".high-unc{color:#E74C3C;font-weight:600;}\n"
            ".med-unc{color:#F39C12;font-weight:600;}\n"
            ".low-unc{color:#27AE60;font-weight:600;}\n"
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
            f"<h1>Scope 3 Uncertainty Analysis &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_total_ci(self, data: Dict[str, Any]) -> str:
        """Render HTML total Scope 3 with 95% CI."""
        ci = data.get("total_scope3_ci", {})
        if not ci:
            return ""
        central = ci.get("central_tco2e")
        lower = ci.get("lower_bound_tco2e")
        upper = ci.get("upper_bound_tco2e")
        unc_pct = ci.get("uncertainty_pct")
        method = ci.get("method", "Monte Carlo simulation")
        confidence = ci.get("confidence_level", "95%")
        cards = [
            ("Central Estimate", _fmt_tco2e(central)),
            ("Lower Bound", _fmt_tco2e(lower)),
            ("Upper Bound", _fmt_tco2e(upper)),
        ]
        if unc_pct is not None:
            cards.append(("Uncertainty", f"+/-{unc_pct:.1f}%"))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Total Scope 3 with 95% Confidence Interval</h2>\n"
            f"<p><strong>Method:</strong> {method} | <strong>Confidence:</strong> {confidence}</p>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_per_category(self, data: Dict[str, Any]) -> str:
        """Render HTML per-category uncertainty ranges."""
        categories = data.get("per_category_uncertainty", [])
        if not categories:
            return ""
        rows = ""
        for cat in sorted(categories, key=lambda x: x.get("uncertainty_pct", 0), reverse=True):
            name = cat.get("category_name", "")
            central = _fmt_tco2e(cat.get("central_tco2e"))
            lower = _fmt_tco2e(cat.get("lower_bound_tco2e"))
            upper = _fmt_tco2e(cat.get("upper_bound_tco2e"))
            unc = cat.get("uncertainty_pct")
            unc_str = f"+/-{unc:.1f}%" if unc is not None else "-"
            unc_css = "high-unc" if unc and unc > 50 else "med-unc" if unc and unc > 25 else "low-unc"
            tier = cat.get("tier", "-")
            rows += (
                f"<tr><td>{name}</td><td>{central}</td><td>{lower}</td>"
                f'<td>{upper}</td><td class="{unc_css}">{unc_str}</td><td>{tier}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>2. Per-Category Uncertainty Ranges</h2>\n"
            "<table><thead><tr><th>Category</th><th>Central</th><th>Lower</th>"
            f"<th>Upper</th><th>Uncertainty</th><th>Tier</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_probability_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML probability distribution histogram."""
        dist = data.get("probability_distribution", {})
        bins = dist.get("bins", [])
        if not bins:
            return ""
        rows = ""
        for b in bins:
            range_str = f"{_fmt_tco2e(b.get('lower'))} - {_fmt_tco2e(b.get('upper'))}"
            freq = b.get("frequency", 0)
            prob = b.get("probability")
            prob_str = f"{prob:.4f}" if prob is not None else "-"
            rows += f"<tr><td>{range_str}</td><td>{freq}</td><td>{prob_str}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>3. Probability Distribution</h2>\n"
            f"<p><strong>Shape:</strong> {dist.get('shape', 'N/A')} | "
            f"<strong>Skewness:</strong> {dist.get('skewness', 'N/A')} | "
            f"<strong>Kurtosis:</strong> {dist.get('kurtosis', 'N/A')}</p>\n"
            "<table><thead><tr><th>Bin Range</th><th>Frequency</th>"
            f"<th>Probability</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis (tornado chart data)."""
        sensitivities = data.get("sensitivity_analysis", [])
        if not sensitivities:
            return ""
        rows = ""
        for s in sorted(sensitivities, key=lambda x: abs(x.get("swing_tco2e", 0)), reverse=True):
            param = s.get("parameter", "")
            low = _fmt_tco2e(s.get("low_case_tco2e"))
            base = _fmt_tco2e(s.get("base_case_tco2e"))
            high = _fmt_tco2e(s.get("high_case_tco2e"))
            swing = _fmt_tco2e(s.get("swing_tco2e"))
            rows += (
                f"<tr><td>{param}</td><td>{low}</td><td>{base}</td>"
                f"<td>{high}</td><td>{swing}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Sensitivity Analysis</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Low</th><th>Base</th>"
            f"<th>High</th><th>Swing</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_tier_correlation(self, data: Dict[str, Any]) -> str:
        """Render HTML tier vs uncertainty correlation."""
        correlation = data.get("tier_correlation", [])
        if not correlation:
            return ""
        rows = ""
        for t in correlation:
            tier = t.get("tier", "")
            avg_unc = t.get("avg_uncertainty_pct")
            avg_str = f"+/-{avg_unc:.1f}%" if avg_unc is not None else "-"
            count = t.get("category_count", 0)
            range_str = t.get("typical_range", "-")
            rows += f"<tr><td>{tier}</td><td>{avg_str}</td><td>{count}</td><td>{range_str}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. Tier vs Uncertainty Correlation</h2>\n"
            "<table><thead><tr><th>Tier</th><th>Avg Uncertainty</th>"
            f"<th>Count</th><th>Range</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_tier_upgrade_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML tier upgrade impact."""
        upgrades = data.get("tier_upgrade_impact", [])
        if not upgrades:
            return ""
        rows = ""
        for u in upgrades:
            cat = u.get("category_name", "")
            current = u.get("current_tier", "-")
            target = u.get("target_tier", "-")
            curr_unc = u.get("current_uncertainty_pct")
            curr_str = f"+/-{curr_unc:.1f}%" if curr_unc is not None else "-"
            proj_unc = u.get("projected_uncertainty_pct")
            proj_str = f"+/-{proj_unc:.1f}%" if proj_unc is not None else "-"
            reduction = u.get("reduction_pct")
            red_str = f"{reduction:.0f}%" if reduction is not None else "-"
            rows += (
                f"<tr><td>{cat}</td><td>{current}</td><td>{target}</td>"
                f"<td>{curr_str}</td><td>{proj_str}</td><td>{red_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Tier Upgrade Impact</h2>\n"
            "<table><thead><tr><th>Category</th><th>Current</th><th>Target</th>"
            "<th>Current Unc</th><th>Projected</th>"
            f"<th>Reduction</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_correlation_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML correlation matrix."""
        matrix = data.get("correlation_matrix", {})
        categories = matrix.get("categories", [])
        values = matrix.get("values", [])
        if not categories or not values:
            return ""
        headers = "".join(f"<th>{c[:10]}</th>" for c in categories)
        rows = ""
        for i, row_vals in enumerate(values):
            cat = categories[i] if i < len(categories) else "?"
            cells = ""
            for val in row_vals:
                if val is not None:
                    bg = ""
                    if abs(val) > 0.7:
                        bg = ' style="background:#fadbd8;"'
                    elif abs(val) > 0.4:
                        bg = ' style="background:#fdebd0;"'
                    cells += f"<td{bg}>{val:.2f}</td>"
                else:
                    cells += "<td>-</td>"
            rows += f"<tr><th>{cat[:10]}</th>{cells}</tr>\n"
        return (
            '<div class="section">\n'
            "<h2>7. Correlation Matrix</h2>\n"
            f"<table><thead><tr><th></th>{headers}</tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
