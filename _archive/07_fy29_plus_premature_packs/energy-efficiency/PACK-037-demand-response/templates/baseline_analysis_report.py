# -*- coding: utf-8 -*-
"""
BaselineAnalysisReportTemplate - CBL methodology comparison for PACK-037.

Generates Customer Baseline Load (CBL) analysis reports comparing projected
baselines under each methodology (e.g., 10-of-10, High-5-of-10, weather-
adjusted regression, meter-before-meter-after). Identifies optimization
opportunities and methodology selection guidance.

Sections:
    1. Baseline Summary
    2. Methodology Comparison
    3. Historical Load Profile
    4. Adjustment Factors
    5. Optimization Opportunities
    6. Methodology Selection Guidance

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - NAESB WEQ Business Practice Standards
    - PJM CBL Methodology (Manual 11)
    - NYISO ICAP Manual (SCR/EDRP baselines)
    - ISO-NE Measurement & Verification Manual

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class BaselineAnalysisReportTemplate:
    """
    CBL methodology comparison report template.

    Renders baseline analysis reports comparing projected Customer Baseline
    Load under multiple methodologies with adjustment factors, optimization
    opportunities, and selection guidance across markdown, HTML, and JSON.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BaselineAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render baseline analysis report as Markdown.

        Args:
            data: Baseline analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_baseline_summary(data),
            self._md_methodology_comparison(data),
            self._md_historical_load_profile(data),
            self._md_adjustment_factors(data),
            self._md_optimization_opportunities(data),
            self._md_methodology_selection(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render baseline analysis report as self-contained HTML.

        Args:
            data: Baseline analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_baseline_summary(data),
            self._html_methodology_comparison(data),
            self._html_historical_load_profile(data),
            self._html_adjustment_factors(data),
            self._html_optimization_opportunities(data),
            self._html_methodology_selection(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Baseline Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render baseline analysis report as structured JSON.

        Args:
            data: Baseline analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "baseline_analysis_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "baseline_summary": self._json_baseline_summary(data),
            "methodology_comparison": data.get("methodology_comparison", []),
            "historical_load_profile": data.get("historical_load_profile", {}),
            "adjustment_factors": data.get("adjustment_factors", []),
            "optimization_opportunities": data.get("optimization_opportunities", []),
            "methodology_selection": data.get("methodology_selection", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Customer Baseline Load Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Meter ID:** {data.get('meter_id', '')}  \n"
            f"**ISO/RTO:** {data.get('iso_rto', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 BaselineAnalysisReportTemplate v37.0.0\n\n---"
        )

    def _md_baseline_summary(self, data: Dict[str, Any]) -> str:
        """Render baseline summary section."""
        summary = data.get("baseline_summary", {})
        return (
            "## 1. Baseline Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Average Daily Peak (kW) | {self._format_power(summary.get('avg_daily_peak_kw', 0))} |\n"
            f"| Methodologies Compared | {summary.get('methodologies_compared', 0)} |\n"
            f"| Recommended Methodology | {summary.get('recommended_methodology', '-')} |\n"
            f"| Baseline Range (Low) | {self._format_power(summary.get('baseline_low_kw', 0))} |\n"
            f"| Baseline Range (High) | {self._format_power(summary.get('baseline_high_kw', 0))} |\n"
            f"| Baseline Spread | {self._fmt(summary.get('baseline_spread_pct', 0))}% |\n"
            f"| Data Quality Score | {self._fmt(summary.get('data_quality_score', 0), 1)}/100 |"
        )

    def _md_methodology_comparison(self, data: Dict[str, Any]) -> str:
        """Render methodology comparison table."""
        methods = data.get("methodology_comparison", [])
        if not methods:
            return "## 2. Methodology Comparison\n\n_No methodology data available._"
        lines = [
            "## 2. Methodology Comparison\n",
            "| Methodology | Baseline kW | Curtailment kW | Revenue Impact | Complexity | Recommended |",
            "|-------------|----------:|---------------:|---------------:|-----------|------------|",
        ]
        for m in methods:
            rec = "Yes" if m.get("recommended", False) else "No"
            lines.append(
                f"| {m.get('methodology', '-')} "
                f"| {self._fmt(m.get('baseline_kw', 0), 1)} "
                f"| {self._fmt(m.get('curtailment_kw', 0), 1)} "
                f"| {self._format_currency(m.get('revenue_impact', 0))} "
                f"| {m.get('complexity', '-')} "
                f"| {rec} |"
            )
        return "\n".join(lines)

    def _md_historical_load_profile(self, data: Dict[str, Any]) -> str:
        """Render historical load profile section."""
        profile = data.get("historical_load_profile", {})
        hourly = profile.get("hourly_averages", [])
        if not hourly:
            return "## 3. Historical Load Profile\n\n_No historical data available._"
        lines = [
            "## 3. Historical Load Profile\n",
            f"**Data Period:** {profile.get('data_period', '-')}  ",
            f"**Data Points:** {profile.get('data_points', 0):,}  ",
            f"**Missing Data:** {self._fmt(profile.get('missing_data_pct', 0))}%\n",
            "| Hour | Avg Load (kW) | Peak Load (kW) | Min Load (kW) |",
            "|-----:|-------------:|---------------:|-------------:|",
        ]
        for h in hourly:
            lines.append(
                f"| {h.get('hour', '-')} "
                f"| {self._fmt(h.get('avg_kw', 0), 1)} "
                f"| {self._fmt(h.get('peak_kw', 0), 1)} "
                f"| {self._fmt(h.get('min_kw', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_adjustment_factors(self, data: Dict[str, Any]) -> str:
        """Render adjustment factors section."""
        factors = data.get("adjustment_factors", [])
        if not factors:
            return "## 4. Adjustment Factors\n\n_No adjustment factors applied._"
        lines = [
            "## 4. Adjustment Factors\n",
            "| Factor | Type | Multiplier | Impact (kW) | Description |",
            "|--------|------|-----------|----------:|-------------|",
        ]
        for f in factors:
            lines.append(
                f"| {f.get('factor', '-')} "
                f"| {f.get('type', '-')} "
                f"| {self._fmt(f.get('multiplier', 1.0), 3)} "
                f"| {self._fmt(f.get('impact_kw', 0), 1)} "
                f"| {f.get('description', '-')} |"
            )
        return "\n".join(lines)

    def _md_optimization_opportunities(self, data: Dict[str, Any]) -> str:
        """Render optimization opportunities section."""
        opps = data.get("optimization_opportunities", [])
        if not opps:
            return "## 5. Optimization Opportunities\n\n_No optimization opportunities identified._"
        lines = [
            "## 5. Optimization Opportunities\n",
            "| # | Opportunity | Baseline Improvement (kW) | Revenue Impact | Effort |",
            "|---|-----------|------------------------:|---------------:|--------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('opportunity', '-')} "
                f"| {self._fmt(o.get('baseline_improvement_kw', 0), 1)} "
                f"| {self._format_currency(o.get('revenue_impact', 0))} "
                f"| {o.get('effort', '-')} |"
            )
        return "\n".join(lines)

    def _md_methodology_selection(self, data: Dict[str, Any]) -> str:
        """Render methodology selection guidance section."""
        selection = data.get("methodology_selection", {})
        lines = [
            "## 6. Methodology Selection Guidance\n",
            f"- **Recommended Methodology:** {selection.get('recommended', '-')}",
            f"- **Rationale:** {selection.get('rationale', '-')}",
            f"- **Revenue Advantage:** {self._format_currency(selection.get('revenue_advantage', 0))}/yr",
            f"- **Risk Considerations:** {selection.get('risk_considerations', '-')}",
            f"- **Implementation Steps:** {selection.get('implementation_steps', '-')}",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Customer Baseline Load Analysis</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'ISO/RTO: {data.get("iso_rto", "-")} | '
            f'Meter: {data.get("meter_id", "-")}</p>'
        )

    def _html_baseline_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline summary cards."""
        s = data.get("baseline_summary", {})
        return (
            '<h2>Baseline Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Avg Daily Peak</span>'
            f'<span class="value">{self._fmt(s.get("avg_daily_peak_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Methods Compared</span>'
            f'<span class="value">{s.get("methodologies_compared", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Recommended</span>'
            f'<span class="value">{s.get("recommended_methodology", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Baseline Spread</span>'
            f'<span class="value">{self._fmt(s.get("baseline_spread_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Data Quality</span>'
            f'<span class="value">{self._fmt(s.get("data_quality_score", 0), 1)}/100</span></div>\n'
            '</div>'
        )

    def _html_methodology_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology comparison table."""
        methods = data.get("methodology_comparison", [])
        rows = ""
        for m in methods:
            cls = "row-recommended" if m.get("recommended", False) else ""
            rows += (
                f'<tr class="{cls}"><td>{m.get("methodology", "-")}</td>'
                f'<td>{self._fmt(m.get("baseline_kw", 0), 1)}</td>'
                f'<td>{self._fmt(m.get("curtailment_kw", 0), 1)}</td>'
                f'<td>{self._format_currency(m.get("revenue_impact", 0))}</td>'
                f'<td>{"Recommended" if m.get("recommended", False) else "-"}</td></tr>\n'
            )
        return (
            '<h2>Methodology Comparison</h2>\n'
            '<table>\n<tr><th>Methodology</th><th>Baseline kW</th>'
            f'<th>Curtailment kW</th><th>Revenue</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_historical_load_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML historical load profile summary."""
        profile = data.get("historical_load_profile", {})
        return (
            '<h2>Historical Load Profile</h2>\n'
            f'<p>Data Period: {profile.get("data_period", "-")} | '
            f'Points: {profile.get("data_points", 0):,} | '
            f'Missing: {self._fmt(profile.get("missing_data_pct", 0))}%</p>'
        )

    def _html_adjustment_factors(self, data: Dict[str, Any]) -> str:
        """Render HTML adjustment factors table."""
        factors = data.get("adjustment_factors", [])
        rows = ""
        for f in factors:
            rows += (
                f'<tr><td>{f.get("factor", "-")}</td>'
                f'<td>{f.get("type", "-")}</td>'
                f'<td>{self._fmt(f.get("multiplier", 1.0), 3)}</td>'
                f'<td>{self._fmt(f.get("impact_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Adjustment Factors</h2>\n'
            '<table>\n<tr><th>Factor</th><th>Type</th>'
            f'<th>Multiplier</th><th>Impact kW</th></tr>\n{rows}</table>'
        )

    def _html_optimization_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML optimization opportunities."""
        opps = data.get("optimization_opportunities", [])
        items = "".join(
            f'<li><strong>{o.get("opportunity", "-")}</strong>: '
            f'+{self._fmt(o.get("baseline_improvement_kw", 0), 1)} kW baseline, '
            f'{self._format_currency(o.get("revenue_impact", 0))} revenue impact</li>\n'
            for o in opps
        )
        return f'<h2>Optimization Opportunities</h2>\n<ol>\n{items}</ol>'

    def _html_methodology_selection(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology selection guidance."""
        sel = data.get("methodology_selection", {})
        return (
            '<h2>Methodology Selection</h2>\n'
            f'<div class="selection-box">'
            f'<strong>Recommended: {sel.get("recommended", "-")}</strong><br>'
            f'Rationale: {sel.get("rationale", "-")}<br>'
            f'Revenue Advantage: {self._format_currency(sel.get("revenue_advantage", 0))}/yr</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_baseline_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON baseline summary."""
        s = data.get("baseline_summary", {})
        return {
            "avg_daily_peak_kw": s.get("avg_daily_peak_kw", 0),
            "methodologies_compared": s.get("methodologies_compared", 0),
            "recommended_methodology": s.get("recommended_methodology", ""),
            "baseline_low_kw": s.get("baseline_low_kw", 0),
            "baseline_high_kw": s.get("baseline_high_kw", 0),
            "baseline_spread_pct": s.get("baseline_spread_pct", 0),
            "data_quality_score": s.get("data_quality_score", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        methods = data.get("methodology_comparison", [])
        hourly = data.get("historical_load_profile", {}).get("hourly_averages", [])
        return {
            "methodology_bar": {
                "type": "bar",
                "labels": [m.get("methodology", "") for m in methods],
                "series": {
                    "baseline_kw": [m.get("baseline_kw", 0) for m in methods],
                    "curtailment_kw": [m.get("curtailment_kw", 0) for m in methods],
                },
            },
            "revenue_comparison": {
                "type": "bar",
                "labels": [m.get("methodology", "") for m in methods],
                "values": [m.get("revenue_impact", 0) for m in methods],
            },
            "load_profile_line": {
                "type": "line",
                "labels": [str(h.get("hour", "")) for h in hourly],
                "series": {
                    "average": [h.get("avg_kw", 0) for h in hourly],
                    "peak": [h.get("peak_kw", 0) for h in hourly],
                    "minimum": [h.get("min_kw", 0) for h in hourly],
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
            ".row-recommended{background:#d1e7dd !important;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".selection-box{background:#d1e7dd;padding:15px;border-radius:8px;margin:10px 0;}"
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

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
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
