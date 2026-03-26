# -*- coding: utf-8 -*-
"""
TimeSeriesDashboard - Multi-Year Trend Chart Data and Consistency for PACK-045.

Generates a time series dashboard covering multi-year emission trends,
consistency analysis between reporting periods, normalized series for
comparisons, and year-over-year variance analysis.

Sections:
    1. Trend Overview
    2. Annual Emissions by Scope
    3. Normalized Series
    4. Consistency Findings
    5. Variance Analysis

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _trend_arrow(current: float, previous: float) -> str:
    """Return trend arrow based on comparison."""
    if current > previous * 1.01:
        return "UP"
    if current < previous * 0.99:
        return "DOWN"
    return "FLAT"


class TimeSeriesDashboard:
    """
    Time series dashboard template.

    Renders multi-year emission trend data, scope-level annual breakdowns,
    normalized intensity series, consistency analysis, and year-over-year
    variance tables. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = TimeSeriesDashboard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TimeSeriesDashboard."""
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
        """Render time series dashboard as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_trend_overview(data),
            self._md_annual_by_scope(data),
            self._md_normalized_series(data),
            self._md_consistency_findings(data),
            self._md_variance_analysis(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render time series dashboard as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_trend_overview(data),
            self._html_annual_by_scope(data),
            self._html_normalized_series(data),
            self._html_consistency_findings(data),
            self._html_variance_analysis(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render time series dashboard as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "time_series_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "years_covered": data.get("years_covered", []),
            "annual_emissions": data.get("annual_emissions", []),
            "normalized_series": data.get("normalized_series", []),
            "consistency_findings": data.get("consistency_findings", []),
            "variance_analysis": data.get("variance_analysis", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        years = data.get("years_covered", [])
        span = f"{years[0]}-{years[-1]}" if years else ""
        return (
            f"# Time Series Dashboard - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Period:** {span} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_trend_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown trend overview."""
        annual = data.get("annual_emissions", [])
        if not annual:
            return "## 1. Trend Overview\n\nNo annual data available."
        first = annual[0]
        last = annual[-1]
        base_total = first.get("total_tco2e", 0)
        current_total = last.get("total_tco2e", 0)
        delta = current_total - base_total
        pct = ((delta / base_total) * 100) if base_total else 0
        return (
            "## 1. Trend Overview\n\n"
            f"- **Base Year ({first.get('year', '')}):** {base_total:,.1f} tCO2e\n"
            f"- **Latest ({last.get('year', '')}):** {current_total:,.1f} tCO2e\n"
            f"- **Change:** {delta:+,.1f} tCO2e ({pct:+.1f}%)\n"
            f"- **Trend:** {_trend_arrow(current_total, base_total)}"
        )

    def _md_annual_by_scope(self, data: Dict[str, Any]) -> str:
        """Render Markdown annual emissions by scope."""
        annual = data.get("annual_emissions", [])
        if not annual:
            return ""
        lines = [
            "## 2. Annual Emissions by Scope",
            "",
            "| Year | Scope 1 | Scope 2 | Scope 3 | Total (tCO2e) |",
            "|------|---------|---------|---------|--------------|",
        ]
        for yr in annual:
            year = yr.get("year", "")
            s1 = yr.get("scope1_tco2e", 0)
            s2 = yr.get("scope2_tco2e", 0)
            s3 = yr.get("scope3_tco2e", 0)
            total = yr.get("total_tco2e", 0)
            lines.append(f"| {year} | {s1:,.1f} | {s2:,.1f} | {s3:,.1f} | {total:,.1f} |")
        return "\n".join(lines)

    def _md_normalized_series(self, data: Dict[str, Any]) -> str:
        """Render Markdown normalized intensity series."""
        series = data.get("normalized_series", [])
        if not series:
            return ""
        lines = [
            "## 3. Normalized Intensity Series",
            "",
            "| Year | Revenue ($M) | tCO2e/Revenue | FTE | tCO2e/FTE | Index (Base=100) |",
            "|------|------------|-------------|-----|----------|-----------------|",
        ]
        for s in series:
            year = s.get("year", "")
            revenue = s.get("revenue_million", 0)
            intensity_rev = s.get("tco2e_per_revenue", 0)
            fte = s.get("fte", 0)
            intensity_fte = s.get("tco2e_per_fte", 0)
            index = s.get("index_base100", 100)
            lines.append(
                f"| {year} | {revenue:,.1f} | {intensity_rev:,.2f} | "
                f"{fte:,} | {intensity_fte:,.2f} | {index:.1f} |"
            )
        return "\n".join(lines)

    def _md_consistency_findings(self, data: Dict[str, Any]) -> str:
        """Render Markdown consistency findings."""
        findings = data.get("consistency_findings", [])
        if not findings:
            return ""
        lines = ["## 4. Consistency Findings", ""]
        for f in findings:
            severity = f.get("severity", "info").upper()
            desc = f.get("description", "")
            years_affected = f.get("years_affected", [])
            yr_str = ", ".join(str(y) for y in years_affected)
            lines.append(f"- **[{severity}]** {desc} (Years: {yr_str})")
        return "\n".join(lines)

    def _md_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown year-over-year variance analysis."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return ""
        lines = [
            "## 5. Year-over-Year Variance",
            "",
            "| Period | Previous (tCO2e) | Current (tCO2e) | Variance | Variance % | Driver |",
            "|--------|-----------------|----------------|----------|-----------|--------|",
        ]
        for v in variances:
            period = v.get("period", "")
            prev = v.get("previous_tco2e", 0)
            curr = v.get("current_tco2e", 0)
            var = curr - prev
            var_pct = ((var / prev) * 100) if prev else 0
            driver = v.get("primary_driver", "")
            lines.append(
                f"| {period} | {prev:,.1f} | {curr:,.1f} | "
                f"{var:+,.1f} | {var_pct:+.1f}% | {driver} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Time Series Dashboard - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".trend-up{color:#e76f51;}\n"
            ".trend-down{color:#2a9d8f;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        years = data.get("years_covered", [])
        span = f"{years[0]}-{years[-1]}" if years else ""
        return (
            '<div class="section">\n'
            f"<h1>Time Series Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Period:</strong> {span}</p>\n<hr>\n</div>"
        )

    def _html_trend_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML trend overview."""
        annual = data.get("annual_emissions", [])
        if not annual:
            return ""
        first = annual[0]
        last = annual[-1]
        base_total = first.get("total_tco2e", 0)
        current_total = last.get("total_tco2e", 0)
        delta = current_total - base_total
        pct = ((delta / base_total) * 100) if base_total else 0
        css = "trend-down" if delta < 0 else "trend-up"
        return (
            '<div class="section">\n<h2>1. Trend Overview</h2>\n'
            f"<p>Base Year: {base_total:,.1f} tCO2e | "
            f"Latest: {current_total:,.1f} tCO2e | "
            f'Change: <span class="{css}">{delta:+,.1f} tCO2e ({pct:+.1f}%)</span></p>\n</div>'
        )

    def _html_annual_by_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML annual emissions by scope table."""
        annual = data.get("annual_emissions", [])
        if not annual:
            return ""
        rows = ""
        for yr in annual:
            year = yr.get("year", "")
            s1 = yr.get("scope1_tco2e", 0)
            s2 = yr.get("scope2_tco2e", 0)
            s3 = yr.get("scope3_tco2e", 0)
            total = yr.get("total_tco2e", 0)
            rows += f"<tr><td>{year}</td><td>{s1:,.1f}</td><td>{s2:,.1f}</td><td>{s3:,.1f}</td><td><strong>{total:,.1f}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Annual Emissions by Scope</h2>\n'
            "<table><thead><tr><th>Year</th><th>Scope 1</th><th>Scope 2</th>"
            "<th>Scope 3</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_normalized_series(self, data: Dict[str, Any]) -> str:
        """Render HTML normalized series table."""
        series = data.get("normalized_series", [])
        if not series:
            return ""
        rows = ""
        for s in series:
            year = s.get("year", "")
            intensity_rev = s.get("tco2e_per_revenue", 0)
            intensity_fte = s.get("tco2e_per_fte", 0)
            index = s.get("index_base100", 100)
            rows += f"<tr><td>{year}</td><td>{intensity_rev:,.2f}</td><td>{intensity_fte:,.2f}</td><td>{index:.1f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Normalized Series</h2>\n'
            "<table><thead><tr><th>Year</th><th>tCO2e/Revenue</th>"
            "<th>tCO2e/FTE</th><th>Index</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_consistency_findings(self, data: Dict[str, Any]) -> str:
        """Render HTML consistency findings."""
        findings = data.get("consistency_findings", [])
        if not findings:
            return ""
        items = ""
        for f in findings:
            severity = f.get("severity", "info").upper()
            desc = f.get("description", "")
            items += f"<li><strong>[{severity}]</strong> {desc}</li>\n"
        return f'<div class="section">\n<h2>4. Consistency Findings</h2>\n<ul>{items}</ul>\n</div>'

    def _html_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML variance analysis table."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return ""
        rows = ""
        for v in variances:
            period = v.get("period", "")
            prev = v.get("previous_tco2e", 0)
            curr = v.get("current_tco2e", 0)
            var = curr - prev
            driver = v.get("primary_driver", "")
            css = "trend-down" if var < 0 else "trend-up"
            rows += (
                f'<tr><td>{period}</td><td>{prev:,.1f}</td><td>{curr:,.1f}</td>'
                f'<td class="{css}">{var:+,.1f}</td><td>{driver}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>5. Variance Analysis</h2>\n'
            "<table><thead><tr><th>Period</th><th>Previous</th><th>Current</th>"
            "<th>Variance</th><th>Driver</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
