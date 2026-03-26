# -*- coding: utf-8 -*-
"""
MultiYearTrendReportTemplate - Base Year Comparison and Trend Analysis for PACK-043.

Generates a multi-year trend report with base year vs subsequent years
comparison table, recalculation history log, methodology-adjusted trend
line, real reduction vs methodology change decomposition, and cumulative
reduction since base year for long-term performance tracking.

Sections:
    1. Base Year Summary
    2. Year-over-Year Comparison Table
    3. Recalculation History Log
    4. Reduction Decomposition (Real vs Methodology)
    5. Cumulative Reduction Since Base Year
    6. Intensity Metrics Trend
    7. Key Insights

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, trend indigo #4A235A theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 43.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"


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


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


class MultiYearTrendReportTemplate:
    """
    Multi-year trend analysis and base year comparison template.

    Renders trend reports with year-over-year comparisons, recalculation
    history, reduction decomposition, and cumulative reduction tracking.
    All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = MultiYearTrendReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MultiYearTrendReportTemplate."""
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
        """Render multi-year trend report as Markdown.

        Args:
            data: Validated trend report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_base_year_summary(data),
            self._md_yoy_comparison(data),
            self._md_recalculation_log(data),
            self._md_decomposition(data),
            self._md_cumulative_reduction(data),
            self._md_intensity_trend(data),
            self._md_key_insights(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render multi-year trend report as HTML.

        Args:
            data: Validated trend report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_base_year_summary(data),
            self._html_yoy_comparison(data),
            self._html_recalculation_log(data),
            self._html_decomposition(data),
            self._html_cumulative_reduction(data),
            self._html_intensity_trend(data),
            self._html_key_insights(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render multi-year trend report as JSON-serializable dict.

        Args:
            data: Validated trend report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "multi_year_trend_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "base_year": data.get("base_year", {}),
            "year_comparison": data.get("year_comparison", []),
            "recalculation_log": data.get("recalculation_log", []),
            "decomposition": data.get("decomposition", []),
            "cumulative_reduction": self._json_cumulative(data),
            "intensity_trend": data.get("intensity_trend", []),
            "key_insights": data.get("key_insights", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Multi-Year Scope 3 Trend Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_base_year_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown base year summary."""
        base = data.get("base_year", {})
        if not base:
            return "## 1. Base Year Summary\n\nNo base year data available."
        lines = [
            "## 1. Base Year Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Base Year | {base.get('year', '-')} |",
            f"| Base Year Scope 3 | {_fmt_tco2e(base.get('scope3_tco2e'))} |",
            f"| Categories Reported | {base.get('categories_reported', '-')} |",
        ]
        methodology = base.get("methodology")
        if methodology:
            lines.append(f"| Methodology | {methodology} |")
        boundary = base.get("organizational_boundary")
        if boundary:
            lines.append(f"| Organizational Boundary | {boundary} |")
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown year-over-year comparison table."""
        years = data.get("year_comparison", [])
        if not years:
            return "## 2. Year-over-Year Comparison\n\nNo comparison data available."
        lines = [
            "## 2. Year-over-Year Comparison",
            "",
            "| Year | Scope 3 tCO2e | vs Base Year | YoY Change | Categories |",
            "|------|-------------|------------|-----------|-----------|",
        ]
        for yr in years:
            year = yr.get("year", "")
            em = _fmt_tco2e(yr.get("scope3_tco2e"))
            vs_base = yr.get("vs_base_year_pct")
            vs_str = _fmt_pct(vs_base) if vs_base is not None else "-"
            yoy = yr.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            cats = yr.get("categories_reported", "-")
            lines.append(f"| {year} | {em} | {vs_str} | {yoy_str} | {cats} |")
        return "\n".join(lines)

    def _md_recalculation_log(self, data: Dict[str, Any]) -> str:
        """Render Markdown recalculation history log."""
        recalcs = data.get("recalculation_log", [])
        if not recalcs:
            return "## 3. Recalculation History\n\nNo recalculations performed."
        lines = [
            "## 3. Recalculation History Log",
            "",
            "| Date | Trigger | Years Affected | Impact (tCO2e) | Description |",
            "|------|---------|---------------|---------------|-------------|",
        ]
        for rc in recalcs:
            date = rc.get("date", "-")
            trigger = rc.get("trigger", "-")
            years_affected = rc.get("years_affected", "-")
            impact = _fmt_tco2e(rc.get("impact_tco2e"))
            desc = rc.get("description", "-")
            lines.append(
                f"| {date} | {trigger} | {years_affected} | {impact} | {desc} |"
            )
        return "\n".join(lines)

    def _md_decomposition(self, data: Dict[str, Any]) -> str:
        """Render Markdown reduction decomposition."""
        decomp = data.get("decomposition", [])
        if not decomp:
            return "## 4. Reduction Decomposition\n\nNo decomposition data available."
        lines = [
            "## 4. Reduction Decomposition (Real vs Methodology)",
            "",
            "| Year | Total Change | Real Reduction | Methodology Change | Structural Change |",
            "|------|------------|---------------|-------------------|-------------------|",
        ]
        for d in decomp:
            year = d.get("year", "")
            total = _fmt_pct(d.get("total_change_pct"))
            real = _fmt_pct(d.get("real_reduction_pct"))
            method = _fmt_pct(d.get("methodology_change_pct"))
            structural = _fmt_pct(d.get("structural_change_pct"))
            lines.append(
                f"| {year} | {total} | {real} | {method} | {structural} |"
            )
        return "\n".join(lines)

    def _md_cumulative_reduction(self, data: Dict[str, Any]) -> str:
        """Render Markdown cumulative reduction since base year."""
        years = data.get("year_comparison", [])
        if not years:
            return "## 5. Cumulative Reduction\n\nNo cumulative data available."
        base = data.get("base_year", {})
        base_em = base.get("scope3_tco2e", 0)
        lines = [
            "## 5. Cumulative Reduction Since Base Year",
            "",
            "| Year | Emissions | Cumulative Reduction | Cumulative % |",
            "|------|----------|---------------------|-------------|",
        ]
        for yr in years:
            year = yr.get("year", "")
            em = yr.get("scope3_tco2e", 0)
            cum_red = base_em - em
            cum_pct = (cum_red / base_em * 100) if base_em > 0 else 0
            lines.append(
                f"| {year} | {_fmt_tco2e(em)} | {_fmt_tco2e(cum_red)} | "
                f"{cum_pct:.1f}% |"
            )
        return "\n".join(lines)

    def _md_intensity_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown intensity metrics trend."""
        intensity = data.get("intensity_trend", [])
        if not intensity:
            return "## 6. Intensity Metrics Trend\n\nNo intensity data available."
        lines = [
            "## 6. Intensity Metrics Trend",
            "",
            "| Year | Revenue ($M) | Intensity (tCO2e/$M) | YoY Change |",
            "|------|------------|---------------------|-----------|",
        ]
        for item in intensity:
            year = item.get("year", "")
            revenue = item.get("revenue_mln")
            rev_str = f"${revenue:.1f}M" if revenue is not None else "-"
            intens = item.get("intensity_tco2e_per_mln")
            int_str = f"{intens:.2f}" if intens is not None else "-"
            yoy = item.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            lines.append(f"| {year} | {rev_str} | {int_str} | {yoy_str} |")
        return "\n".join(lines)

    def _md_key_insights(self, data: Dict[str, Any]) -> str:
        """Render Markdown key insights."""
        insights = data.get("key_insights", [])
        if not insights:
            return "## 7. Key Insights\n\nNo insights available."
        lines = ["## 7. Key Insights", ""]
        for insight in insights:
            lines.append(f"- {insight}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Multi-Year Trend Report - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#4A235A;--primary-light:#7D3C98;--accent:#AF7AC5;"
            "--bg:#F4ECF7;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#F4ECF7;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".positive{color:var(--success);font-weight:700;}\n"
            ".negative{color:var(--danger);font-weight:700;}\n"
            ".insight-list{list-style-type:disc;padding-left:1.5rem;}\n"
            ".insight-list li{margin-bottom:0.5rem;}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Multi-Year Scope 3 Trend Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_base_year_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML base year summary."""
        base = data.get("base_year", {})
        if not base:
            return ""
        rows = (
            f"<tr><td>Base Year</td><td>{base.get('year', '-')}</td></tr>\n"
            f"<tr><td>Base Year Scope 3</td><td>{_fmt_tco2e(base.get('scope3_tco2e'))}</td></tr>\n"
            f"<tr><td>Categories Reported</td><td>{base.get('categories_reported', '-')}</td></tr>\n"
        )
        methodology = base.get("methodology")
        if methodology:
            rows += f"<tr><td>Methodology</td><td>{methodology}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Base Year Summary</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison."""
        years = data.get("year_comparison", [])
        if not years:
            return ""
        rows = ""
        for yr in years:
            year = yr.get("year", "")
            em = _fmt_tco2e(yr.get("scope3_tco2e"))
            vs_base = yr.get("vs_base_year_pct")
            vs_str = _fmt_pct(vs_base) if vs_base is not None else "-"
            yoy = yr.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            cats = yr.get("categories_reported", "-")
            css = "positive" if (vs_base or 0) < 0 else "negative" if (vs_base or 0) > 0 else ""
            rows += (
                f"<tr><td>{year}</td><td>{em}</td>"
                f'<td class="{css}">{vs_str}</td>'
                f"<td>{yoy_str}</td><td>{cats}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Year-over-Year Comparison</h2>\n"
            "<table><thead><tr><th>Year</th><th>Scope 3</th>"
            "<th>vs Base Year</th><th>YoY Change</th><th>Categories</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recalculation_log(self, data: Dict[str, Any]) -> str:
        """Render HTML recalculation log."""
        recalcs = data.get("recalculation_log", [])
        if not recalcs:
            return ""
        rows = ""
        for rc in recalcs:
            date = rc.get("date", "-")
            trigger = rc.get("trigger", "-")
            years_affected = rc.get("years_affected", "-")
            impact = _fmt_tco2e(rc.get("impact_tco2e"))
            desc = rc.get("description", "-")
            rows += (
                f"<tr><td>{date}</td><td>{trigger}</td><td>{years_affected}</td>"
                f"<td>{impact}</td><td>{desc}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Recalculation History Log</h2>\n"
            "<table><thead><tr><th>Date</th><th>Trigger</th>"
            "<th>Years Affected</th><th>Impact</th><th>Description</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML reduction decomposition."""
        decomp = data.get("decomposition", [])
        if not decomp:
            return ""
        rows = ""
        for d in decomp:
            year = d.get("year", "")
            total = _fmt_pct(d.get("total_change_pct"))
            real = _fmt_pct(d.get("real_reduction_pct"))
            method = _fmt_pct(d.get("methodology_change_pct"))
            structural = _fmt_pct(d.get("structural_change_pct"))
            rows += (
                f"<tr><td>{year}</td><td>{total}</td><td>{real}</td>"
                f"<td>{method}</td><td>{structural}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Reduction Decomposition</h2>\n"
            "<table><thead><tr><th>Year</th><th>Total Change</th>"
            "<th>Real Reduction</th><th>Methodology</th>"
            "<th>Structural</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_cumulative_reduction(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative reduction."""
        years = data.get("year_comparison", [])
        base = data.get("base_year", {})
        base_em = base.get("scope3_tco2e", 0)
        if not years or base_em == 0:
            return ""
        rows = ""
        for yr in years:
            year = yr.get("year", "")
            em = yr.get("scope3_tco2e", 0)
            cum_red = base_em - em
            cum_pct = (cum_red / base_em * 100) if base_em > 0 else 0
            css = "positive" if cum_red > 0 else "negative"
            rows += (
                f"<tr><td>{year}</td><td>{_fmt_tco2e(em)}</td>"
                f"<td>{_fmt_tco2e(cum_red)}</td>"
                f'<td class="{css}">{cum_pct:.1f}%</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>5. Cumulative Reduction Since Base Year</h2>\n"
            "<table><thead><tr><th>Year</th><th>Emissions</th>"
            "<th>Cum. Reduction</th><th>Cum. %</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_intensity_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML intensity trend."""
        intensity = data.get("intensity_trend", [])
        if not intensity:
            return ""
        rows = ""
        for item in intensity:
            year = item.get("year", "")
            revenue = item.get("revenue_mln")
            rev_str = f"${revenue:.1f}M" if revenue is not None else "-"
            intens = item.get("intensity_tco2e_per_mln")
            int_str = f"{intens:.2f}" if intens is not None else "-"
            yoy = item.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            rows += (
                f"<tr><td>{year}</td><td>{rev_str}</td>"
                f"<td>{int_str}</td><td>{yoy_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Intensity Metrics Trend</h2>\n"
            "<table><thead><tr><th>Year</th><th>Revenue</th>"
            "<th>Intensity (tCO2e/$M)</th><th>YoY</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_key_insights(self, data: Dict[str, Any]) -> str:
        """Render HTML key insights."""
        insights = data.get("key_insights", [])
        if not insights:
            return ""
        items = "".join(f"<li>{insight}</li>\n" for insight in insights)
        return (
            '<div class="section">\n'
            "<h2>7. Key Insights</h2>\n"
            f'<ul class="insight-list">\n{items}</ul>\n</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_cumulative(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build cumulative reduction chart data."""
        years = data.get("year_comparison", [])
        base = data.get("base_year", {})
        base_em = base.get("scope3_tco2e", 0)
        result = []
        for yr in years:
            em = yr.get("scope3_tco2e", 0)
            cum_red = base_em - em
            cum_pct = (cum_red / base_em * 100) if base_em > 0 else 0
            result.append({
                "year": yr.get("year"),
                "scope3_tco2e": em,
                "cumulative_reduction_tco2e": cum_red,
                "cumulative_reduction_pct": round(cum_pct, 1),
            })
        return result
