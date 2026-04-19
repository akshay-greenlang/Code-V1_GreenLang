# -*- coding: utf-8 -*-
"""
PerformanceReportTemplate - ISO 50001 Clause 9.1 M&V Report for PACK-034.

Generates comprehensive energy performance reports aligned with ISO 50001:2018
Clause 9.1. Covers EnPI dashboard with KPI cards, consumption vs baseline
(normalized), CUSUM status, savings summary, trend analysis, weather
normalization, year-over-year comparison, data quality metrics, and
recommendations.

Sections:
    1. Executive Summary
    2. EnPI Dashboard (KPI cards)
    3. Consumption vs Baseline (normalized)
    4. CUSUM Status
    5. Savings Summary
    6. Trend Analysis
    7. Weather Normalization
    8. Year-over-Year Comparison
    9. Data Quality Metrics
    10. Recommendations

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PerformanceReportTemplate:
    """
    ISO 50001 energy performance M&V report template.

    Renders energy performance monitoring and verification reports with
    EnPI dashboards, CUSUM analysis, normalized consumption comparisons,
    weather normalization, and year-over-year trends across markdown,
    HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PerformanceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render performance report as Markdown.

        Args:
            data: Performance data including enpi_values, cusum_data,
                  savings, trend_data, weather_normalization, and
                  yoy_comparison.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_enpi_dashboard(data),
            self._md_consumption_vs_baseline(data),
            self._md_cusum_status(data),
            self._md_savings_summary(data),
            self._md_trend_analysis(data),
            self._md_weather_normalization(data),
            self._md_yoy_comparison(data),
            self._md_data_quality(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render performance report as self-contained HTML.

        Args:
            data: Performance data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_enpi_dashboard(data),
            self._html_consumption_vs_baseline(data),
            self._html_cusum_status(data),
            self._html_savings_summary(data),
            self._html_trend_analysis(data),
            self._html_yoy_comparison(data),
            self._html_data_quality(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Performance Report - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render performance report as structured JSON.

        Args:
            data: Performance data dict.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "performance_report",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility_name": data.get("facility_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "executive_summary": self._json_executive_summary(data),
            "enpi_values": data.get("enpi_values", []),
            "consumption_vs_baseline": data.get("consumption_vs_baseline", []),
            "cusum_data": data.get("cusum_data", {}),
            "savings": data.get("savings", {}),
            "trend_data": data.get("trend_data", []),
            "weather_normalization": data.get("weather_normalization", {}),
            "yoy_comparison": data.get("yoy_comparison", []),
            "data_quality": data.get("data_quality", {}),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with report metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Energy Performance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**ISO 50001:2018 Clause:** 9.1  \n"
            f"**Report Type:** {data.get('report_type', 'Monthly M&V Report')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 PerformanceReportTemplate v34.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        savings = data.get("savings", {})
        cusum = data.get("cusum_data", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Energy Savings (Period) | {self._format_energy(savings.get('period_savings_mwh', 0))} |\n"
            f"| Cost Savings (Period) | {self._format_currency(savings.get('period_cost_savings', 0))} |\n"
            f"| Cumulative Savings (YTD) | {self._format_energy(savings.get('ytd_savings_mwh', 0))} |\n"
            f"| CUSUM Status | {cusum.get('status', '-')} |\n"
            f"| Overall EnPI Trend | {data.get('overall_enpi_trend', '-')} |\n"
            f"| CO2 Avoided (Period) | {self._fmt(savings.get('co2_avoided_tonnes', 0))} tonnes |"
        )

    def _md_enpi_dashboard(self, data: Dict[str, Any]) -> str:
        """Render EnPI dashboard section with KPI cards."""
        enpis = data.get("enpi_values", [])
        if not enpis:
            return "## 2. EnPI Dashboard\n\n_No EnPI data available._"
        lines = [
            "## 2. EnPI Dashboard\n",
            "| EnPI | Current | Baseline | Target | Change (%) | Status |",
            "|------|---------|----------|--------|-----------|--------|",
        ]
        for e in enpis:
            baseline = e.get("baseline_value", 0)
            current = e.get("current_value", 0)
            change = 0.0
            if baseline and baseline != 0:
                change = ((current - baseline) / abs(baseline)) * 100
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {self._fmt(current)} {e.get('unit', '')} "
                f"| {self._fmt(baseline)} {e.get('unit', '')} "
                f"| {e.get('target', '-')} "
                f"| {self._fmt(change)}% "
                f"| {e.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_consumption_vs_baseline(self, data: Dict[str, Any]) -> str:
        """Render consumption vs baseline (normalized) section."""
        periods = data.get("consumption_vs_baseline", [])
        if not periods:
            return "## 3. Consumption vs Baseline (Normalized)\n\n_No comparison data available._"
        lines = [
            "## 3. Consumption vs Baseline (Normalized)\n",
            "| Period | Actual (MWh) | Baseline Expected (MWh) | Adjusted Baseline (MWh) | Savings (MWh) | Savings (%) |",
            "|--------|-------------|----------------------|----------------------|-------------|------------|",
        ]
        for p in periods:
            actual = p.get("actual_mwh", 0)
            adjusted = p.get("adjusted_baseline_mwh", 0)
            savings_mwh = adjusted - actual
            savings_pct = (savings_mwh / adjusted * 100) if adjusted else 0
            lines.append(
                f"| {p.get('period', '-')} "
                f"| {self._fmt(actual)} "
                f"| {self._fmt(p.get('baseline_expected_mwh', 0))} "
                f"| {self._fmt(adjusted)} "
                f"| {self._fmt(savings_mwh)} "
                f"| {self._fmt(savings_pct)}% |"
            )
        return "\n".join(lines)

    def _md_cusum_status(self, data: Dict[str, Any]) -> str:
        """Render CUSUM status section."""
        cusum = data.get("cusum_data", {})
        points = cusum.get("points", [])
        lines = [
            "## 4. CUSUM Status\n",
            f"**Current CUSUM Value:** {self._format_energy(cusum.get('current_value', 0))}  ",
            f"**CUSUM Trend:** {cusum.get('trend', '-')}  ",
            f"**Status:** {cusum.get('status', '-')}  ",
            f"**Alert Threshold:** {self._format_energy(cusum.get('alert_threshold', 0))}\n",
        ]
        if points:
            lines.extend([
                "### CUSUM Data Points\n",
                "| Period | Savings (MWh) | Cumulative Savings (MWh) |",
                "|--------|-------------|------------------------|",
            ])
            for pt in points:
                lines.append(
                    f"| {pt.get('period', '-')} "
                    f"| {self._fmt(pt.get('savings_mwh', 0))} "
                    f"| {self._fmt(pt.get('cumulative_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render savings summary section."""
        savings = data.get("savings", {})
        by_seu = savings.get("by_seu", [])
        lines = [
            "## 5. Savings Summary\n",
            "| Metric | Period | Year-to-Date | Annual Target |\n"
            "|--------|--------|-------------|---------------|\n"
            f"| Energy Savings | {self._format_energy(savings.get('period_savings_mwh', 0))} "
            f"| {self._format_energy(savings.get('ytd_savings_mwh', 0))} "
            f"| {self._format_energy(savings.get('annual_target_mwh', 0))} |\n"
            f"| Cost Savings | {self._format_currency(savings.get('period_cost_savings', 0))} "
            f"| {self._format_currency(savings.get('ytd_cost_savings', 0))} "
            f"| {self._format_currency(savings.get('annual_target_cost', 0))} |\n"
            f"| CO2 Avoided | {self._fmt(savings.get('co2_avoided_tonnes', 0))} t "
            f"| {self._fmt(savings.get('ytd_co2_avoided', 0))} t "
            f"| {self._fmt(savings.get('annual_target_co2', 0))} t |",
        ]
        if by_seu:
            lines.extend([
                "\n### Savings by SEU\n",
                "| SEU | Energy Saved (MWh) | Cost Saved | Share (%) |",
                "|-----|-------------------|-----------|-----------|",
            ])
            for s in by_seu:
                lines.append(
                    f"| {s.get('seu', '-')} "
                    f"| {self._fmt(s.get('savings_mwh', 0))} "
                    f"| {self._format_currency(s.get('cost_savings', 0))} "
                    f"| {self._fmt(s.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis section."""
        trends = data.get("trend_data", [])
        if not trends:
            return "## 6. Trend Analysis\n\n_No trend data available._"
        lines = [
            "## 6. Trend Analysis\n",
            "| Period | Consumption (MWh) | EnPI Value | Normalized | Trend |",
            "|--------|------------------|-----------|-----------|-------|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt(t.get('consumption_mwh', 0))} "
                f"| {self._fmt(t.get('enpi_value', 0))} "
                f"| {self._fmt(t.get('normalized_value', 0))} "
                f"| {t.get('trend', '-')} |"
            )
        return "\n".join(lines)

    def _md_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render weather normalization section."""
        weather = data.get("weather_normalization", {})
        if not weather:
            return "## 7. Weather Normalization\n\n_No weather normalization data available._"
        lines = [
            "## 7. Weather Normalization\n",
            f"**Normalization Method:** {weather.get('method', 'Degree-day regression')}  ",
            f"**Heating Degree Days (HDD):** {self._fmt(weather.get('hdd', 0), 0)}  ",
            f"**Cooling Degree Days (CDD):** {self._fmt(weather.get('cdd', 0), 0)}  ",
            f"**Base Temperature:** {self._fmt(weather.get('base_temp_c', 18), 1)} C  ",
            f"**Weather Station:** {weather.get('weather_station', '-')}\n",
        ]
        adjustments = weather.get("adjustments", [])
        if adjustments:
            lines.extend([
                "### Monthly Weather Adjustments\n",
                "| Month | HDD | CDD | Actual (MWh) | Weather-Adjusted (MWh) | Adjustment (MWh) |",
                "|-------|-----|-----|-------------|----------------------|-----------------|",
            ])
            for a in adjustments:
                lines.append(
                    f"| {a.get('month', '-')} "
                    f"| {self._fmt(a.get('hdd', 0), 0)} "
                    f"| {self._fmt(a.get('cdd', 0), 0)} "
                    f"| {self._fmt(a.get('actual_mwh', 0))} "
                    f"| {self._fmt(a.get('adjusted_mwh', 0))} "
                    f"| {self._fmt(a.get('adjustment_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render year-over-year comparison section."""
        yoy = data.get("yoy_comparison", [])
        if not yoy:
            return "## 8. Year-over-Year Comparison\n\n_No YoY data available._"
        lines = [
            "## 8. Year-over-Year Comparison\n",
            "| Metric | Current Year | Previous Year | Change | Change (%) |",
            "|--------|-------------|-------------|--------|-----------|",
        ]
        for y in yoy:
            current = y.get("current_year", 0)
            previous = y.get("previous_year", 0)
            change = current - previous
            change_pct = (change / abs(previous) * 100) if previous else 0
            lines.append(
                f"| {y.get('metric', '-')} "
                f"| {self._fmt(current)} {y.get('unit', '')} "
                f"| {self._fmt(previous)} {y.get('unit', '')} "
                f"| {self._fmt(change)} "
                f"| {self._fmt(change_pct)}% |"
            )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality metrics section."""
        dq = data.get("data_quality", {})
        lines = [
            "## 9. Data Quality Metrics\n",
            f"**Data Completeness:** {self._fmt(dq.get('completeness_pct', 0))}%  ",
            f"**Metering Coverage:** {self._fmt(dq.get('metering_coverage_pct', 0))}%  ",
            f"**Estimated Data Share:** {self._fmt(dq.get('estimated_data_pct', 0))}%  ",
            f"**Data Gaps:** {dq.get('data_gaps', 0)} periods  ",
            f"**Overall Quality Score:** {self._fmt(dq.get('overall_score', 0))}%",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 10. Recommendations\n\n_No specific recommendations._"
        lines = ["## 10. Recommendations\n"]
        for i, r in enumerate(recs, 1):
            if isinstance(r, dict):
                lines.append(
                    f"{i}. **{r.get('title', '-')}** - {r.get('description', '-')} "
                    f"(Priority: {r.get('priority', '-')})"
                )
            else:
                lines.append(f"{i}. {r}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Energy Performance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'ISO 50001 Clause 9.1</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        savings = data.get("savings", {})
        cusum = data.get("cusum_data", {})
        return (
            '<h2>1. Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Energy Savings</span>'
            f'<span class="value">{self._format_energy(savings.get("period_savings_mwh", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(savings.get("period_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">CUSUM Status</span>'
            f'<span class="value">{cusum.get("status", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Avoided</span>'
            f'<span class="value">{self._fmt(savings.get("co2_avoided_tonnes", 0))} t</span></div>\n'
            '</div>'
        )

    def _html_enpi_dashboard(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI dashboard."""
        enpis = data.get("enpi_values", [])
        rows = ""
        for e in enpis:
            status = e.get("status", "").lower()
            cls = "status-improved" if "improv" in status else "status-declined" if "declin" in status else ""
            rows += (
                f'<tr><td><strong>{e.get("name", "-")}</strong></td>'
                f'<td>{self._fmt(e.get("current_value", 0))} {e.get("unit", "")}</td>'
                f'<td>{self._fmt(e.get("baseline_value", 0))} {e.get("unit", "")}</td>'
                f'<td>{e.get("target", "-")}</td>'
                f'<td class="{cls}">{e.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. EnPI Dashboard</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Current</th>'
            f'<th>Baseline</th><th>Target</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_consumption_vs_baseline(self, data: Dict[str, Any]) -> str:
        """Render HTML consumption vs baseline."""
        periods = data.get("consumption_vs_baseline", [])
        rows = ""
        for p in periods:
            actual = p.get("actual_mwh", 0)
            adjusted = p.get("adjusted_baseline_mwh", 0)
            savings = adjusted - actual
            cls = "variance-positive" if savings >= 0 else "variance-negative"
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{self._fmt(actual)}</td>'
                f'<td>{self._fmt(adjusted)}</td>'
                f'<td class="{cls}">{self._fmt(savings)}</td></tr>\n'
            )
        return (
            '<h2>3. Consumption vs Baseline</h2>\n'
            '<table>\n<tr><th>Period</th><th>Actual (MWh)</th>'
            f'<th>Adjusted Baseline (MWh)</th><th>Savings (MWh)</th></tr>\n{rows}</table>'
        )

    def _html_cusum_status(self, data: Dict[str, Any]) -> str:
        """Render HTML CUSUM status."""
        cusum = data.get("cusum_data", {})
        status = cusum.get("status", "Neutral")
        cls = "status-improved" if "sav" in status.lower() else "status-declined" if "loss" in status.lower() else ""
        return (
            '<h2>4. CUSUM Status</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CUSUM Value</span>'
            f'<span class="value">{self._format_energy(cusum.get("current_value", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Trend</span>'
            f'<span class="value">{cusum.get("trend", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Status</span>'
            f'<span class="value {cls}">{status}</span></div>\n'
            '</div>'
        )

    def _html_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML savings summary."""
        savings = data.get("savings", {})
        return (
            '<h2>5. Savings Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Period Energy</span>'
            f'<span class="value">{self._format_energy(savings.get("period_savings_mwh", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">YTD Energy</span>'
            f'<span class="value">{self._format_energy(savings.get("ytd_savings_mwh", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Period Cost</span>'
            f'<span class="value">{self._format_currency(savings.get("period_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">YTD Cost</span>'
            f'<span class="value">{self._format_currency(savings.get("ytd_cost_savings", 0))}</span></div>\n'
            '</div>'
        )

    def _html_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML trend analysis."""
        trends = data.get("trend_data", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt(t.get("consumption_mwh", 0))}</td>'
                f'<td>{self._fmt(t.get("enpi_value", 0))}</td>'
                f'<td>{t.get("trend", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Trend Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Consumption (MWh)</th>'
            f'<th>EnPI Value</th><th>Trend</th></tr>\n{rows}</table>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison."""
        yoy = data.get("yoy_comparison", [])
        rows = ""
        for y in yoy:
            current = y.get("current_year", 0)
            previous = y.get("previous_year", 0)
            change = current - previous
            cls = "variance-positive" if change <= 0 else "variance-negative"
            rows += (
                f'<tr><td>{y.get("metric", "-")}</td>'
                f'<td>{self._fmt(current)} {y.get("unit", "")}</td>'
                f'<td>{self._fmt(previous)} {y.get("unit", "")}</td>'
                f'<td class="{cls}">{self._fmt(change)}</td></tr>\n'
            )
        return (
            '<h2>8. Year-over-Year Comparison</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Current Year</th>'
            f'<th>Previous Year</th><th>Change</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality metrics."""
        dq = data.get("data_quality", {})
        return (
            '<h2>9. Data Quality Metrics</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Completeness</span>'
            f'<span class="value">{self._fmt(dq.get("completeness_pct", 0), 0)}%</span></div>\n'
            f'  <div class="card"><span class="label">Metering Coverage</span>'
            f'<span class="value">{self._fmt(dq.get("metering_coverage_pct", 0), 0)}%</span></div>\n'
            f'  <div class="card"><span class="label">Quality Score</span>'
            f'<span class="value">{self._fmt(dq.get("overall_score", 0), 0)}%</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = ""
        for r in recs:
            if isinstance(r, dict):
                items += (
                    f'<li><strong>{r.get("title", "-")}</strong> - '
                    f'{r.get("description", "-")}</li>\n'
                )
            else:
                items += f'<li>{r}</li>\n'
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        savings = data.get("savings", {})
        cusum = data.get("cusum_data", {})
        return {
            "period_savings_mwh": savings.get("period_savings_mwh", 0),
            "period_cost_savings": savings.get("period_cost_savings", 0),
            "ytd_savings_mwh": savings.get("ytd_savings_mwh", 0),
            "cusum_status": cusum.get("status", ""),
            "overall_enpi_trend": data.get("overall_enpi_trend", ""),
            "co2_avoided_tonnes": savings.get("co2_avoided_tonnes", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trends = data.get("trend_data", [])
        cusum_pts = data.get("cusum_data", {}).get("points", [])
        periods = data.get("consumption_vs_baseline", [])
        return {
            "enpi_trend_line": {
                "type": "line",
                "labels": [t.get("period", "") for t in trends],
                "series": {
                    "consumption": [t.get("consumption_mwh", 0) for t in trends],
                    "normalized": [t.get("normalized_value", 0) for t in trends],
                    "enpi": [t.get("enpi_value", 0) for t in trends],
                },
            },
            "cusum_chart": {
                "type": "line",
                "labels": [p.get("period", "") for p in cusum_pts],
                "values": [p.get("cumulative_mwh", 0) for p in cusum_pts],
            },
            "baseline_comparison_bar": {
                "type": "grouped_bar",
                "labels": [p.get("period", "") for p in periods],
                "series": {
                    "actual": [p.get("actual_mwh", 0) for p in periods],
                    "baseline": [p.get("adjusted_baseline_mwh", 0) for p in periods],
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-improved{color:#198754;font-weight:600;}"
            ".status-declined{color:#dc3545;font-weight:600;}"
            ".variance-positive{color:#198754;font-weight:600;}"
            ".variance-negative{color:#dc3545;font-weight:600;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
