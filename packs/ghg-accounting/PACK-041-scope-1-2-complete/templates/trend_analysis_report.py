# -*- coding: utf-8 -*-
"""
TrendAnalysisReportTemplate - Year-over-Year Trend Analysis Report for PACK-041.

Generates a comprehensive trend analysis report covering multi-year emission
summary tables, absolute and percentage changes, Kaya decomposition chart data,
per-factor contribution to change, intensity metrics table (tCO2e per revenue,
FTE, m2, etc.), SBTi trajectory comparison, base year recalculation history,
and weather normalization adjustments.

Sections:
    1. Multi-Year Emission Summary
    2. Absolute and Percentage Changes
    3. Kaya Decomposition
    4. Per-Factor Contribution
    5. Intensity Metrics
    6. SBTi Trajectory Comparison
    7. Base Year Recalculation History
    8. Weather Normalization

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Regulatory References:
    - GHG Protocol Corporate Standard, Ch. 5 (Tracking Over Time)
    - SBTi Criteria and Recommendations v5.1
    - ISO 14064-1:2018 Clause 5.5

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


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_intensity(value: Optional[float]) -> str:
    """Format intensity metric value."""
    if value is None:
        return "N/A"
    return f"{value:,.4f}"


class TrendAnalysisReportTemplate:
    """
    Year-over-year trend analysis report template.

    Renders comprehensive trend analysis reports covering multi-year emissions,
    Kaya decomposition, intensity metrics, SBTi trajectory comparison, base
    year recalculations, and weather normalization. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = TrendAnalysisReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TrendAnalysisReportTemplate.

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
        """Render trend analysis report as Markdown.

        Args:
            data: Validated trend analysis data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_multi_year_summary(data),
            self._md_absolute_pct_changes(data),
            self._md_kaya_decomposition(data),
            self._md_per_factor_contribution(data),
            self._md_intensity_metrics(data),
            self._md_sbti_trajectory(data),
            self._md_base_year_recalculation(data),
            self._md_weather_normalization(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render trend analysis report as HTML.

        Args:
            data: Validated trend analysis data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_multi_year_summary(data),
            self._html_absolute_pct_changes(data),
            self._html_kaya_decomposition(data),
            self._html_intensity_metrics(data),
            self._html_sbti_trajectory(data),
            self._html_base_year_recalculation(data),
            self._html_weather_normalization(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render trend analysis report as JSON-serializable dict.

        Args:
            data: Validated trend analysis data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "trend_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "base_year": self._get_val(data, "base_year"),
            "multi_year_summary": data.get("multi_year_summary", []),
            "yoy_changes": data.get("yoy_changes", []),
            "kaya_decomposition": data.get("kaya_decomposition", {}),
            "per_factor_contribution": data.get("per_factor_contribution", []),
            "intensity_metrics": data.get("intensity_metrics", []),
            "sbti_trajectory": data.get("sbti_trajectory", {}),
            "base_year_recalculations": data.get("base_year_recalculations", []),
            "weather_normalization": data.get("weather_normalization", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        base = self._get_val(data, "base_year")
        base_str = f" | **Base Year:** {base}" if base else ""
        return (
            f"# Trend Analysis Report - {company}\n\n"
            f"**Reporting Year:** {year}{base_str} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_multi_year_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown multi-year emission summary table."""
        summary = data.get("multi_year_summary", [])
        if not summary:
            return "## 1. Multi-Year Emission Summary\n\nNo multi-year data available."
        lines = [
            "## 1. Multi-Year Emission Summary",
            "",
            "| Year | Scope 1 | Scope 2 (Loc) | Scope 2 (Mkt) | Combined (Loc) | Combined (Mkt) |",
            "|------|---------|-------------|-------------|---------------|---------------|",
        ]
        for yr in sorted(summary, key=lambda x: x.get("year", 0)):
            year = yr.get("year", "")
            s1 = _fmt_tco2e(yr.get("scope1_tco2e"))
            s2_loc = _fmt_tco2e(yr.get("scope2_location_tco2e"))
            s2_mkt = _fmt_tco2e(yr.get("scope2_market_tco2e"))
            comb_loc = _fmt_tco2e(yr.get("combined_location_tco2e"))
            comb_mkt = _fmt_tco2e(yr.get("combined_market_tco2e"))
            lines.append(f"| {year} | {s1} | {s2_loc} | {s2_mkt} | {comb_loc} | {comb_mkt} |")
        return "\n".join(lines)

    def _md_absolute_pct_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown absolute and percentage changes."""
        changes = data.get("yoy_changes", [])
        if not changes:
            return "## 2. Year-over-Year Changes\n\nNo YoY change data available."
        lines = [
            "## 2. Year-over-Year Changes",
            "",
            "| Period | Metric | Prior tCO2e | Current tCO2e | Absolute Change | % Change |",
            "|--------|--------|-----------|-------------|----------------|----------|",
        ]
        for chg in changes:
            period = chg.get("period", "")
            metric = chg.get("metric", "")
            prior = _fmt_tco2e(chg.get("prior_tco2e"))
            current = _fmt_tco2e(chg.get("current_tco2e"))
            abs_chg = _fmt_tco2e(chg.get("absolute_change_tco2e"))
            pct_chg = _fmt_pct(chg.get("pct_change"))
            lines.append(f"| {period} | {metric} | {prior} | {current} | {abs_chg} | {pct_chg} |")
        return "\n".join(lines)

    def _md_kaya_decomposition(self, data: Dict[str, Any]) -> str:
        """Render Markdown Kaya decomposition chart data."""
        kaya = data.get("kaya_decomposition", {})
        if not kaya:
            return "## 3. Kaya Decomposition\n\nNo decomposition data available."
        factors = kaya.get("factors", [])
        total_change = kaya.get("total_change_tco2e", 0.0)
        total_pct = kaya.get("total_change_pct", 0.0)
        lines = [
            "## 3. Kaya Decomposition",
            "",
            f"**Total Emission Change:** {_fmt_tco2e(total_change)} ({_fmt_pct(total_pct)})",
            "",
            "| Factor | Description | Contribution tCO2e | Contribution % |",
            "|--------|-----------|-------------------|----------------|",
        ]
        for fac in factors:
            name = fac.get("factor_name", "")
            desc = fac.get("description", "")
            contrib = _fmt_tco2e(fac.get("contribution_tco2e"))
            pct = _fmt_pct(fac.get("contribution_pct"))
            lines.append(f"| {name} | {desc} | {contrib} | {pct} |")
        return "\n".join(lines)

    def _md_per_factor_contribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown per-factor contribution to change."""
        factors = data.get("per_factor_contribution", [])
        if not factors:
            return ""
        lines = [
            "## 4. Per-Factor Contribution to Change",
            "",
            "| Factor | Direction | tCO2e Impact | % of Total Change | Driver |",
            "|--------|----------|-------------|------------------|--------|",
        ]
        for fac in sorted(factors, key=lambda x: abs(x.get("tco2e_impact", 0)), reverse=True):
            name = fac.get("factor_name", "")
            direction = fac.get("direction", "-")
            impact = _fmt_tco2e(fac.get("tco2e_impact"))
            pct = f"{fac.get('pct_of_total_change', 0):.1f}%"
            driver = fac.get("driver", "-")
            lines.append(f"| {name} | {direction} | {impact} | {pct} | {driver} |")
        return "\n".join(lines)

    def _md_intensity_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown intensity metrics table."""
        metrics = data.get("intensity_metrics", [])
        if not metrics:
            return "## 5. Intensity Metrics\n\nNo intensity metrics available."
        lines = [
            "## 5. Intensity Metrics",
            "",
            "| Metric | Denominator | Unit | Current | Prior | % Change | Base Year | Base Year % Change |",
            "|--------|-----------|------|---------|-------|---------|-----------|-------------------|",
        ]
        for m in metrics:
            name = m.get("metric_name", "")
            denom = m.get("denominator", "")
            unit = m.get("unit", "")
            current = _fmt_intensity(m.get("current_value"))
            prior = _fmt_intensity(m.get("prior_value"))
            pct_chg = _fmt_pct(m.get("yoy_change_pct"))
            base = _fmt_intensity(m.get("base_year_value"))
            base_pct = _fmt_pct(m.get("base_year_change_pct"))
            lines.append(f"| {name} | {denom} | {unit} | {current} | {prior} | {pct_chg} | {base} | {base_pct} |")
        return "\n".join(lines)

    def _md_sbti_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi trajectory comparison."""
        sbti = data.get("sbti_trajectory", {})
        if not sbti:
            return "## 6. SBTi Trajectory Comparison\n\nNo SBTi trajectory data."
        targets = sbti.get("targets", [])
        lines = [
            "## 6. SBTi Trajectory Comparison",
            "",
        ]
        for tgt in targets:
            name = tgt.get("target_name", "")
            base_yr = tgt.get("base_year", "")
            target_yr = tgt.get("target_year", "")
            base_em = _fmt_tco2e(tgt.get("base_year_emissions"))
            target_em = _fmt_tco2e(tgt.get("target_emissions"))
            current_em = _fmt_tco2e(tgt.get("current_emissions"))
            required_path = _fmt_tco2e(tgt.get("required_path_emissions"))
            on_track = "Yes" if tgt.get("on_track") else "No"
            gap = _fmt_tco2e(tgt.get("gap_tco2e"))
            lines.extend([
                f"### {name}",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
                f"| Base Year ({base_yr}) | {base_em} |",
                f"| Target Year ({target_yr}) | {target_em} |",
                f"| Current Actual | {current_em} |",
                f"| Required Path | {required_path} |",
                f"| On Track | **{on_track}** |",
                f"| Gap to Path | {gap} |",
                "",
            ])
        # Trajectory data points for chart
        trajectory = sbti.get("trajectory_data", [])
        if trajectory:
            lines.extend([
                "### Trajectory Chart Data",
                "",
                "| Year | Target Path | Actual | Gap |",
                "|------|------------|--------|-----|",
            ])
            for tp in trajectory:
                year = tp.get("year", "")
                path = _fmt_tco2e(tp.get("target_path_tco2e"))
                actual = _fmt_tco2e(tp.get("actual_tco2e"))
                gap = _fmt_tco2e(tp.get("gap_tco2e"))
                lines.append(f"| {year} | {path} | {actual} | {gap} |")
        return "\n".join(lines)

    def _md_base_year_recalculation(self, data: Dict[str, Any]) -> str:
        """Render Markdown base year recalculation history."""
        recalcs = data.get("base_year_recalculations", [])
        if not recalcs:
            return "## 7. Base Year Recalculation History\n\nNo base year recalculations."
        lines = [
            "## 7. Base Year Recalculation History",
            "",
            "| Date | Trigger | Original tCO2e | Revised tCO2e | Change tCO2e | Change % | Approved By |",
            "|------|---------|---------------|-------------|-------------|---------|------------|",
        ]
        for rec in recalcs:
            date = rec.get("date", "")
            trigger = rec.get("trigger", "")
            original = _fmt_tco2e(rec.get("original_tco2e"))
            revised = _fmt_tco2e(rec.get("revised_tco2e"))
            change = _fmt_tco2e(rec.get("change_tco2e"))
            pct = _fmt_pct(rec.get("change_pct"))
            approved = rec.get("approved_by", "-")
            lines.append(f"| {date} | {trigger} | {original} | {revised} | {change} | {pct} | {approved} |")
        return "\n".join(lines)

    def _md_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render Markdown weather normalization section."""
        weather = data.get("weather_normalization", {})
        if not weather:
            return ""
        lines = [
            "## 8. Weather Normalization",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Method | {weather.get('method', 'Heating/Cooling Degree Days')} |",
            f"| HDD Baseline | {weather.get('hdd_baseline', 'N/A')} |",
            f"| CDD Baseline | {weather.get('cdd_baseline', 'N/A')} |",
            f"| Actual HDD | {weather.get('actual_hdd', 'N/A')} |",
            f"| Actual CDD | {weather.get('actual_cdd', 'N/A')} |",
            f"| Normalized Emissions | {_fmt_tco2e(weather.get('normalized_emissions_tco2e'))} |",
            f"| Actual Emissions | {_fmt_tco2e(weather.get('actual_emissions_tco2e'))} |",
            f"| Weather Adjustment | {_fmt_tco2e(weather.get('weather_adjustment_tco2e'))} |",
        ]
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
            f"<title>Trend Analysis - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".increase{color:#e63946;}\n"
            ".decrease{color:#2a9d8f;}\n"
            ".on-track{color:#2a9d8f;font-weight:700;}\n"
            ".off-track{color:#e63946;font-weight:700;}\n"
            ".total-row{font-weight:bold;background:#e8eef4;}\n"
            ".bar-pos{display:inline-block;height:14px;background:#e63946;border-radius:2px;}\n"
            ".bar-neg{display:inline-block;height:14px;background:#2a9d8f;border-radius:2px;}\n"
            ".section{margin-bottom:2rem;}\n"
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
            f"<h1>Trend Analysis Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_multi_year_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML multi-year summary."""
        summary = data.get("multi_year_summary", [])
        if not summary:
            return ""
        rows = ""
        for yr in sorted(summary, key=lambda x: x.get("year", 0)):
            year = yr.get("year", "")
            s1 = _fmt_tco2e(yr.get("scope1_tco2e"))
            s2_loc = _fmt_tco2e(yr.get("scope2_location_tco2e"))
            s2_mkt = _fmt_tco2e(yr.get("scope2_market_tco2e"))
            comb_loc = _fmt_tco2e(yr.get("combined_location_tco2e"))
            rows += f"<tr><td>{year}</td><td>{s1}</td><td>{s2_loc}</td><td>{s2_mkt}</td><td>{comb_loc}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Multi-Year Summary</h2>\n"
            "<table><thead><tr><th>Year</th><th>Scope 1</th><th>S2 (Loc)</th>"
            "<th>S2 (Mkt)</th><th>Combined (Loc)</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_absolute_pct_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML YoY changes."""
        changes = data.get("yoy_changes", [])
        if not changes:
            return ""
        rows = ""
        for chg in changes:
            period = chg.get("period", "")
            metric = chg.get("metric", "")
            pct = chg.get("pct_change")
            css = "increase" if pct and pct > 0 else "decrease"
            abs_chg = _fmt_tco2e(chg.get("absolute_change_tco2e"))
            pct_str = _fmt_pct(pct)
            rows += (
                f"<tr><td>{period}</td><td>{metric}</td>"
                f"<td>{_fmt_tco2e(chg.get('prior_tco2e'))}</td>"
                f"<td>{_fmt_tco2e(chg.get('current_tco2e'))}</td>"
                f'<td>{abs_chg}</td><td class="{css}">{pct_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>2. Year-over-Year Changes</h2>\n"
            "<table><thead><tr><th>Period</th><th>Metric</th><th>Prior</th>"
            "<th>Current</th><th>Abs Change</th><th>% Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_kaya_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML Kaya decomposition."""
        kaya = data.get("kaya_decomposition", {})
        if not kaya:
            return ""
        factors = kaya.get("factors", [])
        rows = ""
        for fac in factors:
            name = fac.get("factor_name", "")
            contrib = fac.get("contribution_tco2e", 0)
            pct = fac.get("contribution_pct", 0)
            bar_class = "bar-pos" if contrib > 0 else "bar-neg"
            bar_width = min(int(abs(pct) * 3), 200)
            rows += (
                f"<tr><td>{name}</td><td>{fac.get('description', '')}</td>"
                f"<td>{_fmt_tco2e(contrib)}</td><td>{_fmt_pct(pct)}</td>"
                f'<td><span class="{bar_class}" style="width:{bar_width}px"></span></td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>3. Kaya Decomposition</h2>\n"
            "<table><thead><tr><th>Factor</th><th>Description</th>"
            "<th>Contribution</th><th>%</th><th>Chart</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_intensity_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML intensity metrics."""
        metrics = data.get("intensity_metrics", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            name = m.get("metric_name", "")
            denom = m.get("denominator", "")
            current = _fmt_intensity(m.get("current_value"))
            prior = _fmt_intensity(m.get("prior_value"))
            pct = m.get("yoy_change_pct")
            css = "increase" if pct and pct > 0 else "decrease"
            pct_str = _fmt_pct(pct)
            rows += (
                f"<tr><td>{name}</td><td>{denom}</td><td>{current}</td>"
                f'<td>{prior}</td><td class="{css}">{pct_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>5. Intensity Metrics</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Denominator</th>"
            "<th>Current</th><th>Prior</th><th>% Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sbti_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi trajectory."""
        sbti = data.get("sbti_trajectory", {})
        if not sbti:
            return ""
        targets = sbti.get("targets", [])
        parts = ['<div class="section">\n<h2>6. SBTi Trajectory</h2>']
        for tgt in targets:
            on_track = tgt.get("on_track")
            css = "on-track" if on_track else "off-track"
            label = "On Track" if on_track else "Off Track"
            parts.append(
                f"<h3>{tgt.get('target_name', '')}</h3>\n"
                "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
                f"<tr><td>Base Year Emissions</td><td>{_fmt_tco2e(tgt.get('base_year_emissions'))}</td></tr>\n"
                f"<tr><td>Current Actual</td><td>{_fmt_tco2e(tgt.get('current_emissions'))}</td></tr>\n"
                f"<tr><td>Required Path</td><td>{_fmt_tco2e(tgt.get('required_path_emissions'))}</td></tr>\n"
                f'<tr><td>Status</td><td class="{css}">{label}</td></tr>\n'
                "</tbody></table>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _html_base_year_recalculation(self, data: Dict[str, Any]) -> str:
        """Render HTML base year recalculation history."""
        recalcs = data.get("base_year_recalculations", [])
        if not recalcs:
            return ""
        rows = ""
        for rec in recalcs:
            date = rec.get("date", "")
            trigger = rec.get("trigger", "")
            original = _fmt_tco2e(rec.get("original_tco2e"))
            revised = _fmt_tco2e(rec.get("revised_tco2e"))
            pct = _fmt_pct(rec.get("change_pct"))
            rows += f"<tr><td>{date}</td><td>{trigger}</td><td>{original}</td><td>{revised}</td><td>{pct}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>7. Base Year Recalculations</h2>\n"
            "<table><thead><tr><th>Date</th><th>Trigger</th><th>Original</th>"
            "<th>Revised</th><th>% Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render HTML weather normalization."""
        weather = data.get("weather_normalization", {})
        if not weather:
            return ""
        return (
            '<div class="section">\n'
            "<h2>8. Weather Normalization</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Method</td><td>{weather.get('method', 'HDD/CDD')}</td></tr>\n"
            f"<tr><td>Normalized Emissions</td><td>{_fmt_tco2e(weather.get('normalized_emissions_tco2e'))}</td></tr>\n"
            f"<tr><td>Actual Emissions</td><td>{_fmt_tco2e(weather.get('actual_emissions_tco2e'))}</td></tr>\n"
            f"<tr><td>Weather Adjustment</td><td>{_fmt_tco2e(weather.get('weather_adjustment_tco2e'))}</td></tr>\n"
            "</tbody></table>\n</div>"
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
