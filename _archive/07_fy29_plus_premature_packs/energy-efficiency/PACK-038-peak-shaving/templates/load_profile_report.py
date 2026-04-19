# -*- coding: utf-8 -*-
"""
LoadProfileReportTemplate - Load profile analysis for PACK-038.

Generates comprehensive load profile analysis reports showing facility
demand duration curves, load factor analysis, day-type consumption
patterns (weekday/weekend/holiday), seasonal profile decomposition,
and anomaly identification with root-cause summary.

Sections:
    1. Profile Summary
    2. Duration Curve Analysis
    3. Load Factor Analysis
    4. Day-Type Patterns
    5. Seasonal Profiles
    6. Anomaly Summary
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IEC 61968 (CIM for load profiles)
    - EN 50160 (voltage quality and load characteristics)
    - ASHRAE Guideline 14 (measurement of energy and demand savings)

Author: GreenLang Team
Version: 38.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class LoadProfileReportTemplate:
    """
    Load profile analysis report template.

    Renders facility load profile analysis reports showing demand duration
    curves, load factor metrics, day-type consumption patterns, seasonal
    decomposition, and anomaly detection across markdown, HTML, and JSON
    formats. All outputs include SHA-256 provenance hashing for audit
    trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LoadProfileReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render load profile report as Markdown.

        Args:
            data: Load profile engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_profile_summary(data),
            self._md_duration_curve(data),
            self._md_load_factor(data),
            self._md_day_type_patterns(data),
            self._md_seasonal_profiles(data),
            self._md_anomaly_summary(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render load profile report as self-contained HTML.

        Args:
            data: Load profile engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_profile_summary(data),
            self._html_duration_curve(data),
            self._html_load_factor(data),
            self._html_day_type_patterns(data),
            self._html_seasonal_profiles(data),
            self._html_anomaly_summary(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Load Profile Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render load profile report as structured JSON.

        Args:
            data: Load profile engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "load_profile_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "profile_summary": self._json_profile_summary(data),
            "duration_curve": data.get("duration_curve", []),
            "load_factor": self._json_load_factor(data),
            "day_type_patterns": data.get("day_type_patterns", []),
            "seasonal_profiles": data.get("seasonal_profiles", []),
            "anomaly_summary": data.get("anomaly_summary", []),
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
        """Render markdown header with facility metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Load Profile Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Meter Points:** {data.get('meter_points', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 LoadProfileReportTemplate v38.0.0\n\n---"
        )

    def _md_profile_summary(self, data: Dict[str, Any]) -> str:
        """Render profile summary section."""
        summary = data.get("profile_summary", {})
        peak_kw = data.get("peak_demand_kw", 0)
        avg_kw = summary.get("average_demand_kw", 0)
        load_factor = self._pct(avg_kw, peak_kw) if peak_kw > 0 else "0.0%"
        return (
            "## 1. Profile Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Peak Demand | {self._format_power(peak_kw)} |\n"
            f"| Average Demand | {self._format_power(avg_kw)} |\n"
            f"| Minimum Demand | {self._format_power(summary.get('minimum_demand_kw', 0))} |\n"
            f"| Load Factor | {load_factor} |\n"
            f"| Total Consumption | {self._format_energy(summary.get('total_consumption_mwh', 0))} |\n"
            f"| Data Points | {self._fmt(summary.get('data_points', 0), 0)} |\n"
            f"| Data Quality Score | {self._fmt(summary.get('data_quality_score', 0), 1)}% |\n"
            f"| Peak-to-Valley Ratio | {self._fmt(summary.get('peak_to_valley_ratio', 0), 2)} |"
        )

    def _md_duration_curve(self, data: Dict[str, Any]) -> str:
        """Render duration curve analysis section."""
        curve = data.get("duration_curve", [])
        if not curve:
            return "## 2. Duration Curve Analysis\n\n_No duration curve data available._"
        lines = [
            "## 2. Duration Curve Analysis\n",
            "| Percentile | Demand (kW) | Hours Above | % of Peak |",
            "|-----------|----------:|----------:|----------:|",
        ]
        for point in curve:
            lines.append(
                f"| {point.get('percentile', '-')} "
                f"| {self._fmt(point.get('demand_kw', 0), 1)} "
                f"| {self._fmt(point.get('hours_above', 0), 0)} "
                f"| {self._fmt(point.get('pct_of_peak', 0))}% |"
            )
        return "\n".join(lines)

    def _md_load_factor(self, data: Dict[str, Any]) -> str:
        """Render load factor analysis section."""
        factors = data.get("load_factor", {})
        return (
            "## 3. Load Factor Analysis\n\n"
            "| Period | Load Factor | Avg kW | Peak kW |\n"
            "|--------|----------:|-------:|--------:|\n"
            f"| Overall | {self._fmt(factors.get('overall', 0))}% "
            f"| {self._fmt(factors.get('overall_avg_kw', 0), 1)} "
            f"| {self._fmt(factors.get('overall_peak_kw', 0), 1)} |\n"
            f"| On-Peak | {self._fmt(factors.get('on_peak', 0))}% "
            f"| {self._fmt(factors.get('on_peak_avg_kw', 0), 1)} "
            f"| {self._fmt(factors.get('on_peak_peak_kw', 0), 1)} |\n"
            f"| Off-Peak | {self._fmt(factors.get('off_peak', 0))}% "
            f"| {self._fmt(factors.get('off_peak_avg_kw', 0), 1)} "
            f"| {self._fmt(factors.get('off_peak_peak_kw', 0), 1)} |\n"
            f"| Shoulder | {self._fmt(factors.get('shoulder', 0))}% "
            f"| {self._fmt(factors.get('shoulder_avg_kw', 0), 1)} "
            f"| {self._fmt(factors.get('shoulder_peak_kw', 0), 1)} |"
        )

    def _md_day_type_patterns(self, data: Dict[str, Any]) -> str:
        """Render day-type consumption patterns section."""
        patterns = data.get("day_type_patterns", [])
        if not patterns:
            return "## 4. Day-Type Patterns\n\n_No day-type pattern data available._"
        lines = [
            "## 4. Day-Type Patterns\n",
            "| Day Type | Avg Demand (kW) | Peak Demand (kW) | Load Factor | Count |",
            "|----------|---------------:|----------------:|----------:|------:|",
        ]
        for pat in patterns:
            lines.append(
                f"| {pat.get('day_type', '-')} "
                f"| {self._fmt(pat.get('avg_demand_kw', 0), 1)} "
                f"| {self._fmt(pat.get('peak_demand_kw', 0), 1)} "
                f"| {self._fmt(pat.get('load_factor', 0))}% "
                f"| {pat.get('count', 0)} |"
            )
        return "\n".join(lines)

    def _md_seasonal_profiles(self, data: Dict[str, Any]) -> str:
        """Render seasonal profile decomposition section."""
        profiles = data.get("seasonal_profiles", [])
        if not profiles:
            return "## 5. Seasonal Profiles\n\n_No seasonal profile data available._"
        lines = [
            "## 5. Seasonal Profiles\n",
            "| Season | Peak kW | Avg kW | Load Factor | Consumption (MWh) |",
            "|--------|-------:|------:|----------:|------------------:|",
        ]
        for prof in profiles:
            lines.append(
                f"| {prof.get('season', '-')} "
                f"| {self._fmt(prof.get('peak_kw', 0), 1)} "
                f"| {self._fmt(prof.get('avg_kw', 0), 1)} "
                f"| {self._fmt(prof.get('load_factor', 0))}% "
                f"| {self._fmt(prof.get('consumption_mwh', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_anomaly_summary(self, data: Dict[str, Any]) -> str:
        """Render anomaly summary section."""
        anomalies = data.get("anomaly_summary", [])
        if not anomalies:
            return "## 6. Anomaly Summary\n\n_No anomalies detected in the analysis period._"
        lines = [
            "## 6. Anomaly Summary\n",
            "| # | Timestamp | Type | Observed kW | Expected kW | Deviation | Severity |",
            "|---|-----------|------|----------:|----------:|----------:|----------|",
        ]
        for i, anomaly in enumerate(anomalies, 1):
            lines.append(
                f"| {i} | {anomaly.get('timestamp', '-')} "
                f"| {anomaly.get('type', '-')} "
                f"| {self._fmt(anomaly.get('observed_kw', 0), 1)} "
                f"| {self._fmt(anomaly.get('expected_kw', 0), 1)} "
                f"| {self._fmt(anomaly.get('deviation_pct', 0))}% "
                f"| {anomaly.get('severity', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Implement demand-side management to improve load factor",
                "Install interval metering for finer granularity profiles",
                "Investigate anomalous demand spikes for corrective action",
                "Target peak shaving during high-demand seasonal windows",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-038 Peak Shaving Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Load Profile Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peak Demand: {self._format_power(data.get("peak_demand_kw", 0))} | '
            f'Period: {data.get("analysis_period", "-")}</p>'
        )

    def _html_profile_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML profile summary cards."""
        s = data.get("profile_summary", {})
        peak = data.get("peak_demand_kw", 0)
        avg = s.get("average_demand_kw", 0)
        lf = self._pct(avg, peak) if peak > 0 else "0.0%"
        return (
            '<h2>Profile Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Peak Demand</span>'
            f'<span class="value">{self._fmt(peak, 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Avg Demand</span>'
            f'<span class="value">{self._fmt(avg, 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Load Factor</span>'
            f'<span class="value">{lf}</span></div>\n'
            f'  <div class="card"><span class="label">Total Consumption</span>'
            f'<span class="value">{self._fmt(s.get("total_consumption_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Data Quality</span>'
            f'<span class="value">{self._fmt(s.get("data_quality_score", 0), 1)}%</span></div>\n'
            '</div>'
        )

    def _html_duration_curve(self, data: Dict[str, Any]) -> str:
        """Render HTML duration curve table."""
        curve = data.get("duration_curve", [])
        rows = ""
        for point in curve:
            rows += (
                f'<tr><td>{point.get("percentile", "-")}</td>'
                f'<td>{self._fmt(point.get("demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(point.get("hours_above", 0), 0)}</td>'
                f'<td>{self._fmt(point.get("pct_of_peak", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Duration Curve Analysis</h2>\n'
            '<table>\n<tr><th>Percentile</th><th>Demand (kW)</th>'
            f'<th>Hours Above</th><th>% of Peak</th></tr>\n{rows}</table>'
        )

    def _html_load_factor(self, data: Dict[str, Any]) -> str:
        """Render HTML load factor analysis."""
        factors = data.get("load_factor", {})
        return (
            '<h2>Load Factor Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Load Factor</th>'
            '<th>Avg kW</th><th>Peak kW</th></tr>\n'
            f'<tr><td>Overall</td><td>{self._fmt(factors.get("overall", 0))}%</td>'
            f'<td>{self._fmt(factors.get("overall_avg_kw", 0), 1)}</td>'
            f'<td>{self._fmt(factors.get("overall_peak_kw", 0), 1)}</td></tr>\n'
            f'<tr><td>On-Peak</td><td>{self._fmt(factors.get("on_peak", 0))}%</td>'
            f'<td>{self._fmt(factors.get("on_peak_avg_kw", 0), 1)}</td>'
            f'<td>{self._fmt(factors.get("on_peak_peak_kw", 0), 1)}</td></tr>\n'
            f'<tr><td>Off-Peak</td><td>{self._fmt(factors.get("off_peak", 0))}%</td>'
            f'<td>{self._fmt(factors.get("off_peak_avg_kw", 0), 1)}</td>'
            f'<td>{self._fmt(factors.get("off_peak_peak_kw", 0), 1)}</td></tr>\n'
            '</table>'
        )

    def _html_day_type_patterns(self, data: Dict[str, Any]) -> str:
        """Render HTML day-type patterns table."""
        patterns = data.get("day_type_patterns", [])
        rows = ""
        for pat in patterns:
            rows += (
                f'<tr><td>{pat.get("day_type", "-")}</td>'
                f'<td>{self._fmt(pat.get("avg_demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(pat.get("peak_demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(pat.get("load_factor", 0))}%</td>'
                f'<td>{pat.get("count", 0)}</td></tr>\n'
            )
        return (
            '<h2>Day-Type Patterns</h2>\n'
            '<table>\n<tr><th>Day Type</th><th>Avg kW</th>'
            f'<th>Peak kW</th><th>Load Factor</th><th>Count</th></tr>\n{rows}</table>'
        )

    def _html_seasonal_profiles(self, data: Dict[str, Any]) -> str:
        """Render HTML seasonal profiles table."""
        profiles = data.get("seasonal_profiles", [])
        rows = ""
        for prof in profiles:
            rows += (
                f'<tr><td>{prof.get("season", "-")}</td>'
                f'<td>{self._fmt(prof.get("peak_kw", 0), 1)}</td>'
                f'<td>{self._fmt(prof.get("avg_kw", 0), 1)}</td>'
                f'<td>{self._fmt(prof.get("load_factor", 0))}%</td>'
                f'<td>{self._fmt(prof.get("consumption_mwh", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Seasonal Profiles</h2>\n'
            '<table>\n<tr><th>Season</th><th>Peak kW</th><th>Avg kW</th>'
            f'<th>Load Factor</th><th>Consumption (MWh)</th></tr>\n{rows}</table>'
        )

    def _html_anomaly_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML anomaly summary table."""
        anomalies = data.get("anomaly_summary", [])
        rows = ""
        for anomaly in anomalies:
            sev = anomaly.get("severity", "low").lower()
            rows += (
                f'<tr><td>{anomaly.get("timestamp", "-")}</td>'
                f'<td>{anomaly.get("type", "-")}</td>'
                f'<td>{self._fmt(anomaly.get("observed_kw", 0), 1)}</td>'
                f'<td>{self._fmt(anomaly.get("expected_kw", 0), 1)}</td>'
                f'<td class="severity-{sev}">{self._fmt(anomaly.get("deviation_pct", 0))}%</td>'
                f'<td class="severity-{sev}">{anomaly.get("severity", "-")}</td></tr>\n'
            )
        return (
            '<h2>Anomaly Summary</h2>\n'
            '<table>\n<tr><th>Timestamp</th><th>Type</th><th>Observed kW</th>'
            f'<th>Expected kW</th><th>Deviation</th><th>Severity</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Implement demand-side management to improve load factor",
            "Install interval metering for finer granularity profiles",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_profile_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON profile summary."""
        s = data.get("profile_summary", {})
        return {
            "peak_demand_kw": data.get("peak_demand_kw", 0),
            "average_demand_kw": s.get("average_demand_kw", 0),
            "minimum_demand_kw": s.get("minimum_demand_kw", 0),
            "total_consumption_mwh": s.get("total_consumption_mwh", 0),
            "data_points": s.get("data_points", 0),
            "data_quality_score": s.get("data_quality_score", 0),
            "peak_to_valley_ratio": s.get("peak_to_valley_ratio", 0),
        }

    def _json_load_factor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON load factor analysis."""
        factors = data.get("load_factor", {})
        return {
            "overall": factors.get("overall", 0),
            "on_peak": factors.get("on_peak", 0),
            "off_peak": factors.get("off_peak", 0),
            "shoulder": factors.get("shoulder", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        curve = data.get("duration_curve", [])
        patterns = data.get("day_type_patterns", [])
        profiles = data.get("seasonal_profiles", [])
        anomalies = data.get("anomaly_summary", [])
        return {
            "duration_curve": {
                "type": "line",
                "labels": [p.get("percentile", "") for p in curve],
                "values": [p.get("demand_kw", 0) for p in curve],
            },
            "day_type_comparison": {
                "type": "grouped_bar",
                "labels": [p.get("day_type", "") for p in patterns],
                "series": {
                    "avg_demand": [p.get("avg_demand_kw", 0) for p in patterns],
                    "peak_demand": [p.get("peak_demand_kw", 0) for p in patterns],
                },
            },
            "seasonal_profile": {
                "type": "bar",
                "labels": [p.get("season", "") for p in profiles],
                "values": [p.get("peak_kw", 0) for p in profiles],
            },
            "anomaly_scatter": {
                "type": "scatter",
                "items": [
                    {
                        "timestamp": a.get("timestamp", ""),
                        "deviation_pct": a.get("deviation_pct", 0),
                        "severity": a.get("severity", ""),
                    }
                    for a in anomalies
                ],
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
            ".severity-high{color:#dc3545;font-weight:700;}"
            ".severity-medium{color:#fd7e14;font-weight:600;}"
            ".severity-low{color:#198754;font-weight:500;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string (e.g., 'EUR 1,234.00').
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string (e.g., '1,234.0 kW').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 MWh').
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

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
