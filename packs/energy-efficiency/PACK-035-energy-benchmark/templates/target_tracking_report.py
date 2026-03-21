# -*- coding: utf-8 -*-
"""
TargetTrackingReportTemplate - Target tracking and trajectory report for PACK-035.

Generates target tracking reports that compare current energy performance
against defined targets (e.g., 20% EUI reduction by 2030), provide
trajectory analysis, variance breakdowns, forecasts to target year,
risk assessments, and corrective action recommendations.

Sections:
    1. Header
    2. Baseline Summary
    3. Target Definition
    4. Current Performance vs Target
    5. Trajectory Chart Data
    6. Variance Analysis
    7. Forecast to Target Year
    8. Risk Assessment
    9. Corrective Actions
   10. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TargetTrackingReportTemplate:
    """
    Target tracking and trajectory report template.

    Renders target tracking reports with baseline summaries, trajectory
    analysis, variance breakdowns, forecasts, risk assessments, and
    corrective actions across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TargetTrackingReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render target tracking report as Markdown.

        Args:
            data: Target tracking data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_baseline(data),
            self._md_target_definition(data),
            self._md_current_vs_target(data),
            self._md_trajectory(data),
            self._md_variance_analysis(data),
            self._md_forecast(data),
            self._md_risk_assessment(data),
            self._md_corrective_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render target tracking report as self-contained HTML.

        Args:
            data: Target tracking data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_baseline(data),
            self._html_target_definition(data),
            self._html_current_vs_target(data),
            self._html_trajectory(data),
            self._html_variance_analysis(data),
            self._html_forecast(data),
            self._html_risk_assessment(data),
            self._html_corrective_actions(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Target Tracking Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render target tracking report as structured JSON.

        Args:
            data: Target tracking data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "target_tracking_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "baseline": data.get("baseline", {}),
            "target": data.get("target", {}),
            "current_performance": data.get("current_performance", {}),
            "trajectory": data.get("trajectory", []),
            "variance_analysis": data.get("variance_analysis", {}),
            "forecast": data.get("forecast", {}),
            "risk_assessment": data.get("risk_assessment", []),
            "corrective_actions": data.get("corrective_actions", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
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
            "# Energy Target Tracking Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Target:** {data.get('target_name', '-')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 TargetTrackingReportTemplate v35.0.0\n\n---"
        )

    def _md_baseline(self, data: Dict[str, Any]) -> str:
        """Render baseline summary section."""
        bl = data.get("baseline", {})
        return (
            "## 1. Baseline Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Year | {bl.get('year', '-')} |\n"
            f"| Baseline EUI | {self._fmt(bl.get('eui_kwh_m2', 0))} kWh/m2/yr |\n"
            f"| Baseline Energy | {self._fmt(bl.get('total_energy_kwh', 0), 0)} kWh/yr |\n"
            f"| Baseline CO2 | {self._fmt(bl.get('co2_kg', 0), 0)} kg CO2/yr |\n"
            f"| Normalisation | {bl.get('normalisation', 'Weather + Occupancy')} |\n"
            f"| Data Quality | {bl.get('data_quality', '-')} |"
        )

    def _md_target_definition(self, data: Dict[str, Any]) -> str:
        """Render target definition section."""
        t = data.get("target", {})
        return (
            "## 2. Target Definition\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Target Name | {t.get('name', '-')} |\n"
            f"| Target Type | {t.get('type', 'Absolute Reduction')} |\n"
            f"| Reduction Target | {self._fmt(t.get('reduction_pct', 0))}% |\n"
            f"| Target EUI | {self._fmt(t.get('target_eui', 0))} kWh/m2/yr |\n"
            f"| Base Year | {t.get('base_year', '-')} |\n"
            f"| Target Year | {t.get('target_year', '-')} |\n"
            f"| Interim Milestones | {t.get('milestones_count', 0)} |\n"
            f"| Aligned Framework | {t.get('framework', '-')} |"
        )

    def _md_current_vs_target(self, data: Dict[str, Any]) -> str:
        """Render current performance vs target section."""
        cp = data.get("current_performance", {})
        return (
            "## 3. Current Performance vs Target\n\n"
            "| Metric | Baseline | Target | Current | Progress |\n"
            "|--------|---------|--------|---------|----------|\n"
            f"| EUI (kWh/m2/yr) | {self._fmt(cp.get('baseline_eui', 0))} "
            f"| {self._fmt(cp.get('target_eui', 0))} "
            f"| {self._fmt(cp.get('current_eui', 0))} "
            f"| {self._fmt(cp.get('eui_progress_pct', 0))}% |\n"
            f"| Total Energy (kWh/yr) | {self._fmt(cp.get('baseline_energy', 0), 0)} "
            f"| {self._fmt(cp.get('target_energy', 0), 0)} "
            f"| {self._fmt(cp.get('current_energy', 0), 0)} "
            f"| {self._fmt(cp.get('energy_progress_pct', 0))}% |\n"
            f"| CO2 (kg/yr) | {self._fmt(cp.get('baseline_co2', 0), 0)} "
            f"| {self._fmt(cp.get('target_co2', 0), 0)} "
            f"| {self._fmt(cp.get('current_co2', 0), 0)} "
            f"| {self._fmt(cp.get('co2_progress_pct', 0))}% |\n\n"
            f"**On Track:** {'Yes' if cp.get('on_track', False) else 'No'}  \n"
            f"**Years Remaining:** {cp.get('years_remaining', '-')}  \n"
            f"**Required Annual Reduction:** {self._fmt(cp.get('required_annual_reduction_pct', 0))}%/yr"
        )

    def _md_trajectory(self, data: Dict[str, Any]) -> str:
        """Render trajectory chart data section."""
        traj = data.get("trajectory", [])
        if not traj:
            return "## 4. Trajectory\n\n_No trajectory data available._"
        lines = [
            "## 4. Trajectory\n",
            "| Year | Planned EUI | Actual EUI | Variance | Status |",
            "|------|-----------|-----------|----------|--------|",
        ]
        for t in traj:
            lines.append(
                f"| {t.get('year', '-')} "
                f"| {self._fmt(t.get('planned_eui', 0))} "
                f"| {self._fmt(t.get('actual_eui', 0))} "
                f"| {self._fmt(t.get('variance', 0))} "
                f"| {t.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render variance analysis section."""
        va = data.get("variance_analysis", {})
        factors = va.get("factors", [])
        lines = [
            "## 5. Variance Analysis\n",
            f"**Total Variance:** {self._fmt(va.get('total_variance', 0))} kWh/m2/yr  ",
            f"**Variance Direction:** {va.get('direction', '-')}  ",
            f"**Primary Cause:** {va.get('primary_cause', '-')}\n",
        ]
        if factors:
            lines.extend([
                "| Factor | Contribution (kWh/m2) | Share (%) |",
                "|--------|----------------------|-----------|",
            ])
            for f in factors:
                lines.append(
                    f"| {f.get('factor', '-')} "
                    f"| {self._fmt(f.get('contribution', 0))} "
                    f"| {self._fmt(f.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_forecast(self, data: Dict[str, Any]) -> str:
        """Render forecast to target year section."""
        fc = data.get("forecast", {})
        scenarios = fc.get("scenarios", [])
        lines = [
            "## 6. Forecast to Target Year\n",
            f"**Forecast Method:** {fc.get('method', 'Linear Extrapolation')}  ",
            f"**Projected EUI at Target Year:** {self._fmt(fc.get('projected_eui', 0))} kWh/m2/yr  ",
            f"**Target EUI:** {self._fmt(fc.get('target_eui', 0))} kWh/m2/yr  ",
            f"**Projected Gap at Target:** {self._fmt(fc.get('projected_gap', 0))} kWh/m2/yr  ",
            f"**Probability of Meeting Target:** {self._fmt(fc.get('probability_pct', 0))}%",
        ]
        if scenarios:
            lines.extend([
                "\n### Scenario Analysis\n",
                "| Scenario | Projected EUI | Gap | Probability |",
                "|----------|-------------|-----|------------|",
            ])
            for s in scenarios:
                lines.append(
                    f"| {s.get('name', '-')} "
                    f"| {self._fmt(s.get('projected_eui', 0))} "
                    f"| {self._fmt(s.get('gap', 0))} "
                    f"| {self._fmt(s.get('probability_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        risks = data.get("risk_assessment", [])
        if not risks:
            return "## 7. Risk Assessment\n\n_No risks identified._"
        lines = [
            "## 7. Risk Assessment\n",
            "| Risk | Likelihood | Impact | Mitigation |",
            "|------|-----------|--------|------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render corrective actions section."""
        actions = data.get("corrective_actions", [])
        if not actions:
            return "## 8. Corrective Actions\n\n_No corrective actions required._"
        lines = [
            "## 8. Corrective Actions\n",
            "| # | Action | Category | Expected Impact | Timeline | Owner |",
            "|---|--------|---------|----------------|----------|-------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('category', '-')} "
                f"| {a.get('expected_impact', '-')} "
                f"| {a.get('timeline', '-')} "
                f"| {a.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Energy Target Tracking Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Target: {data.get("target_name", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_baseline(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline summary."""
        bl = data.get("baseline", {})
        return (
            '<h2>Baseline Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Baseline Year</span>'
            f'<span class="value">{bl.get("year", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Baseline EUI</span>'
            f'<span class="value">{self._fmt(bl.get("eui_kwh_m2", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card"><span class="label">Baseline Energy</span>'
            f'<span class="value">{self._fmt(bl.get("total_energy_kwh", 0), 0)}</span>'
            f'<span class="label">kWh/yr</span></div>\n'
            '</div>'
        )

    def _html_target_definition(self, data: Dict[str, Any]) -> str:
        """Render HTML target definition."""
        t = data.get("target", {})
        return (
            '<h2>Target Definition</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>{t.get("name", "-")}</strong> | '
            f'Reduction: {self._fmt(t.get("reduction_pct", 0))}% | '
            f'Target EUI: {self._fmt(t.get("target_eui", 0))} kWh/m2/yr | '
            f'By: {t.get("target_year", "-")}</p></div>'
        )

    def _html_current_vs_target(self, data: Dict[str, Any]) -> str:
        """Render HTML current performance vs target."""
        cp = data.get("current_performance", {})
        on_track = cp.get("on_track", False)
        status_cls = "status-pass" if on_track else "status-fail"
        status_text = "ON TRACK" if on_track else "OFF TRACK"
        progress = cp.get("eui_progress_pct", 0)
        bar_color = "#198754" if on_track else "#dc3545"
        return (
            '<h2>Current Performance vs Target</h2>\n'
            f'<p class="{status_cls}" style="font-size:1.3em;">{status_text} | '
            f'Progress: {self._fmt(progress)}%</p>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{min(progress, 100)}%;'
            f'background:{bar_color};"></div></div>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Current EUI</span>'
            f'<span class="value">{self._fmt(cp.get("current_eui", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Target EUI</span>'
            f'<span class="value">{self._fmt(cp.get("target_eui", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Years Left</span>'
            f'<span class="value">{cp.get("years_remaining", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Req. Annual Cut</span>'
            f'<span class="value">{self._fmt(cp.get("required_annual_reduction_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML trajectory table."""
        traj = data.get("trajectory", [])
        rows = ""
        for t in traj:
            status = t.get("status", "")
            cls = "status-pass" if status == "ON_TRACK" else ("status-fail" if status == "OFF_TRACK" else "")
            rows += (
                f'<tr><td>{t.get("year", "-")}</td>'
                f'<td>{self._fmt(t.get("planned_eui", 0))}</td>'
                f'<td>{self._fmt(t.get("actual_eui", 0))}</td>'
                f'<td>{self._fmt(t.get("variance", 0))}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            '<h2>Trajectory</h2>\n'
            '<table>\n<tr><th>Year</th><th>Planned EUI</th><th>Actual EUI</th>'
            f'<th>Variance</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML variance analysis."""
        va = data.get("variance_analysis", {})
        factors = va.get("factors", [])
        rows = "".join(
            f'<tr><td>{f.get("factor", "-")}</td>'
            f'<td>{self._fmt(f.get("contribution", 0))}</td>'
            f'<td>{self._fmt(f.get("share_pct", 0))}%</td></tr>\n'
            for f in factors
        )
        return (
            '<h2>Variance Analysis</h2>\n'
            f'<p>Total Variance: <strong>{self._fmt(va.get("total_variance", 0))} kWh/m2/yr</strong> | '
            f'Cause: {va.get("primary_cause", "-")}</p>\n'
            '<table>\n<tr><th>Factor</th><th>Contribution</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML forecast section."""
        fc = data.get("forecast", {})
        prob = fc.get("probability_pct", 0)
        prob_cls = "status-pass" if prob >= 70 else ("status-warn" if prob >= 40 else "status-fail")
        return (
            '<h2>Forecast to Target Year</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Projected EUI:</strong> {self._fmt(fc.get("projected_eui", 0))} kWh/m2/yr | '
            f'<strong>Target:</strong> {self._fmt(fc.get("target_eui", 0))} kWh/m2/yr | '
            f'<strong>Gap:</strong> {self._fmt(fc.get("projected_gap", 0))} kWh/m2/yr</p>'
            f'<p class="{prob_cls}">Probability of Meeting Target: '
            f'{self._fmt(prob)}%</p></div>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment table."""
        risks = data.get("risk_assessment", [])
        rows = ""
        for r in risks:
            impact = r.get("impact", "Low").lower()
            cls = "risk-high" if impact == "high" else ("risk-medium" if impact == "medium" else "risk-low")
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td class="{cls}">{r.get("impact", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Risk Assessment</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML corrective actions."""
        actions = data.get("corrective_actions", [])
        items = "".join(
            f'<li><strong>{a.get("action", "-")}</strong> '
            f'({a.get("category", "-")}) | '
            f'Impact: {a.get("expected_impact", "-")} | '
            f'By: {a.get("timeline", "-")} | '
            f'Owner: {a.get("owner", "-")}</li>\n'
            for a in actions
        )
        return f'<h2>Corrective Actions</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        traj = data.get("trajectory", [])
        factors = data.get("variance_analysis", {}).get("factors", [])
        return {
            "trajectory_line": {
                "type": "line",
                "labels": [t.get("year", "") for t in traj],
                "series": {
                    "planned": [t.get("planned_eui", 0) for t in traj],
                    "actual": [t.get("actual_eui", 0) for t in traj],
                },
            },
            "variance_waterfall": {
                "type": "waterfall",
                "labels": [f.get("factor", "") for f in factors],
                "values": [f.get("contribution", 0) for f in factors],
            },
            "progress_gauge": {
                "type": "gauge",
                "value": data.get("current_performance", {}).get("eui_progress_pct", 0),
                "target": 100,
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
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".progress-bar{height:24px;background:#e9ecef;border-radius:6px;margin:15px 0;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;transition:width 0.3s;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".status-warn{color:#fd7e14;font-weight:700;}"
            ".risk-high{color:#dc3545;font-weight:700;}"
            ".risk-medium{color:#fd7e14;font-weight:600;}"
            ".risk-low{color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
