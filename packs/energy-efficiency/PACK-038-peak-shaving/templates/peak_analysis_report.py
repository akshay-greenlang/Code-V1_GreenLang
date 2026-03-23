# -*- coding: utf-8 -*-
"""
PeakAnalysisReportTemplate - Peak identification analysis for PACK-038.

Generates comprehensive peak identification reports showing top-N demand
peak ranking, peak attribution breakdown by load category, clustering
analysis of coincident peaks, avoidability assessment with shaving
potential estimates for each identified peak event.

Sections:
    1. Peak Overview
    2. Top-N Peak Ranking
    3. Attribution Breakdown
    4. Clustering Analysis
    5. Avoidability Assessment
    6. Shaving Potential
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - NERC reliability standards (coincident peak definitions)
    - PJM/ERCOT/ISO-NE peak calculation methodologies
    - EU Network Code on Demand Connection (Art. 28)

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


class PeakAnalysisReportTemplate:
    """
    Peak identification and analysis report template.

    Renders peak demand analysis reports showing top-N peak ranking,
    attribution by load category, clustering of coincident peaks,
    avoidability scoring, and shaving potential estimates across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PeakAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render peak analysis report as Markdown.

        Args:
            data: Peak analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_peak_overview(data),
            self._md_top_n_ranking(data),
            self._md_attribution_breakdown(data),
            self._md_clustering_analysis(data),
            self._md_avoidability_assessment(data),
            self._md_shaving_potential(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render peak analysis report as self-contained HTML.

        Args:
            data: Peak analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_peak_overview(data),
            self._html_top_n_ranking(data),
            self._html_attribution_breakdown(data),
            self._html_clustering_analysis(data),
            self._html_avoidability_assessment(data),
            self._html_shaving_potential(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Peak Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render peak analysis report as structured JSON.

        Args:
            data: Peak analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "peak_analysis_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "peak_overview": self._json_peak_overview(data),
            "top_n_ranking": data.get("top_n_ranking", []),
            "attribution_breakdown": data.get("attribution_breakdown", []),
            "clustering_analysis": data.get("clustering_analysis", []),
            "avoidability_assessment": data.get("avoidability_assessment", []),
            "shaving_potential": self._json_shaving_potential(data),
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
            f"# Peak Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Utility/ISO:** {data.get('utility_iso', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 PeakAnalysisReportTemplate v38.0.0\n\n---"
        )

    def _md_peak_overview(self, data: Dict[str, Any]) -> str:
        """Render peak overview summary section."""
        overview = data.get("peak_overview", {})
        return (
            "## 1. Peak Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Annual Peak Demand | {self._format_power(overview.get('annual_peak_kw', 0))} |\n"
            f"| Coincident Peak (CP) | {self._format_power(overview.get('coincident_peak_kw', 0))} |\n"
            f"| Non-Coincident Peak (NCP) | {self._format_power(overview.get('non_coincident_peak_kw', 0))} |\n"
            f"| Peaks Analyzed | {overview.get('peaks_analyzed', 0)} |\n"
            f"| Avoidable Peaks | {overview.get('avoidable_peaks', 0)} |\n"
            f"| Total Shavable kW | {self._format_power(overview.get('total_shavable_kw', 0))} |\n"
            f"| Estimated Annual Savings | {self._format_currency(overview.get('estimated_savings', 0))} |\n"
            f"| Peak Concentration Index | {self._fmt(overview.get('peak_concentration_index', 0), 2)} |"
        )

    def _md_top_n_ranking(self, data: Dict[str, Any]) -> str:
        """Render top-N peak ranking table."""
        peaks = data.get("top_n_ranking", [])
        if not peaks:
            return "## 2. Top-N Peak Ranking\n\n_No peak ranking data available._"
        lines = [
            "## 2. Top-N Peak Ranking\n",
            "| Rank | Date/Time | Demand (kW) | Type | Duration (min) | Avoidable |",
            "|-----:|-----------|----------:|------|-------------:|-----------|",
        ]
        for i, peak in enumerate(peaks, 1):
            avoidable = "Yes" if peak.get("avoidable", False) else "No"
            lines.append(
                f"| {i} | {peak.get('timestamp', '-')} "
                f"| {self._fmt(peak.get('demand_kw', 0), 1)} "
                f"| {peak.get('peak_type', '-')} "
                f"| {peak.get('duration_min', 0)} "
                f"| {avoidable} |"
            )
        return "\n".join(lines)

    def _md_attribution_breakdown(self, data: Dict[str, Any]) -> str:
        """Render peak attribution breakdown section."""
        attrs = data.get("attribution_breakdown", [])
        if not attrs:
            return "## 3. Attribution Breakdown\n\n_No attribution data available._"
        lines = [
            "## 3. Attribution Breakdown\n",
            "| Load Category | Contribution (kW) | Share (%) | Coincidence Factor |",
            "|--------------|------------------:|----------:|------------------:|",
        ]
        for attr in attrs:
            lines.append(
                f"| {attr.get('category', '-')} "
                f"| {self._fmt(attr.get('contribution_kw', 0), 1)} "
                f"| {self._fmt(attr.get('share_pct', 0))}% "
                f"| {self._fmt(attr.get('coincidence_factor', 0), 3)} |"
            )
        return "\n".join(lines)

    def _md_clustering_analysis(self, data: Dict[str, Any]) -> str:
        """Render clustering analysis section."""
        clusters = data.get("clustering_analysis", [])
        if not clusters:
            return "## 4. Clustering Analysis\n\n_No clustering data available._"
        lines = [
            "## 4. Clustering Analysis\n",
            "| Cluster | Peak Count | Avg Demand (kW) | Common Hours | Primary Driver |",
            "|---------|----------:|---------------:|-------------|---------------|",
        ]
        for cl in clusters:
            lines.append(
                f"| {cl.get('cluster_id', '-')} "
                f"| {cl.get('peak_count', 0)} "
                f"| {self._fmt(cl.get('avg_demand_kw', 0), 1)} "
                f"| {cl.get('common_hours', '-')} "
                f"| {cl.get('primary_driver', '-')} |"
            )
        return "\n".join(lines)

    def _md_avoidability_assessment(self, data: Dict[str, Any]) -> str:
        """Render avoidability assessment section."""
        assessments = data.get("avoidability_assessment", [])
        if not assessments:
            return "## 5. Avoidability Assessment\n\n_No avoidability data available._"
        lines = [
            "## 5. Avoidability Assessment\n",
            "| Peak # | Demand (kW) | Avoidable kW | Strategy | Confidence |",
            "|-------:|----------:|------------:|----------|----------:|",
        ]
        for i, assess in enumerate(assessments, 1):
            lines.append(
                f"| {i} | {self._fmt(assess.get('demand_kw', 0), 1)} "
                f"| {self._fmt(assess.get('avoidable_kw', 0), 1)} "
                f"| {assess.get('strategy', '-')} "
                f"| {self._fmt(assess.get('confidence_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_shaving_potential(self, data: Dict[str, Any]) -> str:
        """Render shaving potential summary section."""
        potential = data.get("shaving_potential", {})
        if not potential:
            return "## 6. Shaving Potential\n\n_No shaving potential data available._"
        return (
            "## 6. Shaving Potential\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Target Peak Reduction | {self._format_power(potential.get('target_reduction_kw', 0))} |\n"
            f"| Achievable Reduction | {self._format_power(potential.get('achievable_reduction_kw', 0))} |\n"
            f"| BESS Contribution | {self._format_power(potential.get('bess_contribution_kw', 0))} |\n"
            f"| Load Shifting Contribution | {self._format_power(potential.get('load_shifting_kw', 0))} |\n"
            f"| DR Contribution | {self._format_power(potential.get('dr_contribution_kw', 0))} |\n"
            f"| New Peak After Shaving | {self._format_power(potential.get('new_peak_kw', 0))} |\n"
            f"| Demand Charge Savings | {self._format_currency(potential.get('demand_charge_savings', 0))} |\n"
            f"| Annual Net Savings | {self._format_currency(potential.get('annual_net_savings', 0))} |"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Deploy BESS to shave top 10 coincident peaks",
                "Implement pre-cooling strategy to reduce HVAC coincidence",
                "Install real-time peak monitoring with automated alerts",
                "Evaluate load shifting for process loads during peak windows",
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
            f'<h1>Peak Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peak Demand: {self._format_power(data.get("peak_demand_kw", 0))} | '
            f'Utility/ISO: {data.get("utility_iso", "-")}</p>'
        )

    def _html_peak_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML peak overview summary cards."""
        o = data.get("peak_overview", {})
        return (
            '<h2>Peak Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Peak</span>'
            f'<span class="value">{self._fmt(o.get("annual_peak_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Coincident Peak</span>'
            f'<span class="value">{self._fmt(o.get("coincident_peak_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Avoidable Peaks</span>'
            f'<span class="value">{o.get("avoidable_peaks", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Shavable kW</span>'
            f'<span class="value">{self._fmt(o.get("total_shavable_kw", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Est. Savings</span>'
            f'<span class="value">{self._format_currency(o.get("estimated_savings", 0))}</span></div>\n'
            '</div>'
        )

    def _html_top_n_ranking(self, data: Dict[str, Any]) -> str:
        """Render HTML top-N peak ranking table."""
        peaks = data.get("top_n_ranking", [])
        rows = ""
        for i, peak in enumerate(peaks, 1):
            avoidable = "Yes" if peak.get("avoidable", False) else "No"
            rows += (
                f'<tr><td>{i}</td><td>{peak.get("timestamp", "-")}</td>'
                f'<td>{self._fmt(peak.get("demand_kw", 0), 1)}</td>'
                f'<td>{peak.get("peak_type", "-")}</td>'
                f'<td>{peak.get("duration_min", 0)}</td>'
                f'<td>{avoidable}</td></tr>\n'
            )
        return (
            '<h2>Top-N Peak Ranking</h2>\n'
            '<table>\n<tr><th>Rank</th><th>Date/Time</th><th>Demand (kW)</th>'
            f'<th>Type</th><th>Duration (min)</th><th>Avoidable</th></tr>\n{rows}</table>'
        )

    def _html_attribution_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML attribution breakdown table."""
        attrs = data.get("attribution_breakdown", [])
        rows = ""
        for attr in attrs:
            rows += (
                f'<tr><td>{attr.get("category", "-")}</td>'
                f'<td>{self._fmt(attr.get("contribution_kw", 0), 1)}</td>'
                f'<td>{self._fmt(attr.get("share_pct", 0))}%</td>'
                f'<td>{self._fmt(attr.get("coincidence_factor", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>Attribution Breakdown</h2>\n'
            '<table>\n<tr><th>Category</th><th>Contribution (kW)</th>'
            f'<th>Share</th><th>Coincidence Factor</th></tr>\n{rows}</table>'
        )

    def _html_clustering_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML clustering analysis table."""
        clusters = data.get("clustering_analysis", [])
        rows = ""
        for cl in clusters:
            rows += (
                f'<tr><td>{cl.get("cluster_id", "-")}</td>'
                f'<td>{cl.get("peak_count", 0)}</td>'
                f'<td>{self._fmt(cl.get("avg_demand_kw", 0), 1)}</td>'
                f'<td>{cl.get("common_hours", "-")}</td>'
                f'<td>{cl.get("primary_driver", "-")}</td></tr>\n'
            )
        return (
            '<h2>Clustering Analysis</h2>\n'
            '<table>\n<tr><th>Cluster</th><th>Count</th><th>Avg kW</th>'
            f'<th>Common Hours</th><th>Primary Driver</th></tr>\n{rows}</table>'
        )

    def _html_avoidability_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML avoidability assessment table."""
        assessments = data.get("avoidability_assessment", [])
        rows = ""
        for i, assess in enumerate(assessments, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{self._fmt(assess.get("demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(assess.get("avoidable_kw", 0), 1)}</td>'
                f'<td>{assess.get("strategy", "-")}</td>'
                f'<td>{self._fmt(assess.get("confidence_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Avoidability Assessment</h2>\n'
            '<table>\n<tr><th>#</th><th>Demand (kW)</th><th>Avoidable kW</th>'
            f'<th>Strategy</th><th>Confidence</th></tr>\n{rows}</table>'
        )

    def _html_shaving_potential(self, data: Dict[str, Any]) -> str:
        """Render HTML shaving potential summary."""
        p = data.get("shaving_potential", {})
        return (
            '<h2>Shaving Potential</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Target Reduction</span>'
            f'<span class="value">{self._fmt(p.get("target_reduction_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Achievable</span>'
            f'<span class="value">{self._fmt(p.get("achievable_reduction_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">New Peak</span>'
            f'<span class="value">{self._fmt(p.get("new_peak_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Demand Savings</span>'
            f'<span class="value">{self._format_currency(p.get("demand_charge_savings", 0))}</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Deploy BESS to shave top coincident peaks",
            "Implement pre-cooling strategy for HVAC loads",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_peak_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON peak overview."""
        o = data.get("peak_overview", {})
        return {
            "annual_peak_kw": o.get("annual_peak_kw", 0),
            "coincident_peak_kw": o.get("coincident_peak_kw", 0),
            "non_coincident_peak_kw": o.get("non_coincident_peak_kw", 0),
            "peaks_analyzed": o.get("peaks_analyzed", 0),
            "avoidable_peaks": o.get("avoidable_peaks", 0),
            "total_shavable_kw": o.get("total_shavable_kw", 0),
            "estimated_savings": o.get("estimated_savings", 0),
            "peak_concentration_index": o.get("peak_concentration_index", 0),
        }

    def _json_shaving_potential(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON shaving potential."""
        p = data.get("shaving_potential", {})
        return {
            "target_reduction_kw": p.get("target_reduction_kw", 0),
            "achievable_reduction_kw": p.get("achievable_reduction_kw", 0),
            "bess_contribution_kw": p.get("bess_contribution_kw", 0),
            "load_shifting_kw": p.get("load_shifting_kw", 0),
            "dr_contribution_kw": p.get("dr_contribution_kw", 0),
            "new_peak_kw": p.get("new_peak_kw", 0),
            "demand_charge_savings": p.get("demand_charge_savings", 0),
            "annual_net_savings": p.get("annual_net_savings", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        peaks = data.get("top_n_ranking", [])
        attrs = data.get("attribution_breakdown", [])
        clusters = data.get("clustering_analysis", [])
        assessments = data.get("avoidability_assessment", [])
        return {
            "peak_ranking_bar": {
                "type": "bar",
                "labels": [p.get("timestamp", "") for p in peaks],
                "values": [p.get("demand_kw", 0) for p in peaks],
            },
            "attribution_pie": {
                "type": "pie",
                "labels": [a.get("category", "") for a in attrs],
                "values": [a.get("contribution_kw", 0) for a in attrs],
            },
            "cluster_scatter": {
                "type": "scatter",
                "items": [
                    {
                        "cluster": c.get("cluster_id", ""),
                        "count": c.get("peak_count", 0),
                        "avg_kw": c.get("avg_demand_kw", 0),
                    }
                    for c in clusters
                ],
            },
            "avoidability_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": a.get("strategy", ""), "value": a.get("avoidable_kw", 0)}
                    for a in assessments
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
