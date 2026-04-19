# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - C-suite summary for PACK-039.

Generates concise 2-4 page executive summary reports for C-suite
audiences showing energy KPI dashboard, cost trend analysis, EnPI
performance highlights, top anomalies requiring attention, and
prioritized recommendations with estimated ROI.

Sections:
    1. Key Performance Indicators
    2. Cost Trends
    3. EnPI Performance
    4. Top Anomalies
    5. Savings Achieved
    6. Action Items
    7. Outlook

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Management review requirements)
    - GRI 302 (Energy disclosure)
    - CDP Climate Change (Energy metrics)

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template.

    Renders concise 2-4 page C-suite executive summary reports showing
    energy KPI dashboard, cost trend analysis, EnPI performance highlights,
    top anomalies, savings achieved, and prioritized action items across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary report as Markdown.

        Args:
            data: Executive summary engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpis(data),
            self._md_cost_trends(data),
            self._md_enpi_performance(data),
            self._md_top_anomalies(data),
            self._md_savings_achieved(data),
            self._md_action_items(data),
            self._md_outlook(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary report as self-contained HTML.

        Args:
            data: Executive summary engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpis(data),
            self._html_cost_trends(data),
            self._html_enpi_performance(data),
            self._html_top_anomalies(data),
            self._html_savings_achieved(data),
            self._html_action_items(data),
            self._html_outlook(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Summary Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary report as structured JSON.

        Args:
            data: Executive summary engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_summary_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpis": self._json_kpis(data),
            "cost_trends": data.get("cost_trends", []),
            "enpi_performance": data.get("enpi_performance", []),
            "top_anomalies": data.get("top_anomalies", []),
            "savings_achieved": self._json_savings(data),
            "action_items": data.get("action_items", []),
            "outlook": data.get("outlook", {}),
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
            f"# Energy Monitoring Executive Summary\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Prepared For:** {data.get('prepared_for', 'Management')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 ExecutiveSummaryReportTemplate v39.0.0\n\n---"
        )

    def _md_kpis(self, data: Dict[str, Any]) -> str:
        """Render key performance indicators section."""
        kpis = data.get("kpis", {})
        return (
            "## 1. Key Performance Indicators\n\n"
            "| KPI | Current | Previous | Change | Target | Status |\n"
            "|-----|--------:|---------:|-------:|-------:|--------|\n"
            f"| Total Consumption (MWh) | {self._fmt(kpis.get('consumption_mwh', 0), 1)} "
            f"| {self._fmt(kpis.get('prev_consumption_mwh', 0), 1)} "
            f"| {self._fmt(kpis.get('consumption_change_pct', 0))}% "
            f"| {self._fmt(kpis.get('target_consumption_mwh', 0), 1)} "
            f"| {kpis.get('consumption_status', '-')} |\n"
            f"| Total Cost | {self._format_currency(kpis.get('total_cost', 0))} "
            f"| {self._format_currency(kpis.get('prev_total_cost', 0))} "
            f"| {self._fmt(kpis.get('cost_change_pct', 0))}% "
            f"| {self._format_currency(kpis.get('target_cost', 0))} "
            f"| {kpis.get('cost_status', '-')} |\n"
            f"| EnPI (kWh/unit) | {self._fmt(kpis.get('enpi_value', 0), 3)} "
            f"| {self._fmt(kpis.get('prev_enpi_value', 0), 3)} "
            f"| {self._fmt(kpis.get('enpi_change_pct', 0))}% "
            f"| {self._fmt(kpis.get('target_enpi', 0), 3)} "
            f"| {kpis.get('enpi_status', '-')} |\n"
            f"| Peak Demand (kW) | {self._fmt(kpis.get('peak_demand_kw', 0), 0)} "
            f"| {self._fmt(kpis.get('prev_peak_demand_kw', 0), 0)} "
            f"| {self._fmt(kpis.get('peak_change_pct', 0))}% "
            f"| {self._fmt(kpis.get('target_peak_kw', 0), 0)} "
            f"| {kpis.get('peak_status', '-')} |\n"
            f"| Carbon Emissions (tCO2) | {self._fmt(kpis.get('carbon_tco2', 0), 1)} "
            f"| {self._fmt(kpis.get('prev_carbon_tco2', 0), 1)} "
            f"| {self._fmt(kpis.get('carbon_change_pct', 0))}% "
            f"| {self._fmt(kpis.get('target_carbon_tco2', 0), 1)} "
            f"| {kpis.get('carbon_status', '-')} |"
        )

    def _md_cost_trends(self, data: Dict[str, Any]) -> str:
        """Render cost trends section."""
        trends = data.get("cost_trends", [])
        if not trends:
            return "## 2. Cost Trends\n\n_No cost trend data available._"
        lines = [
            "## 2. Cost Trends\n",
            "| Period | Energy Cost | Demand Cost | Total | vs Budget | vs Prior Year |",
            "|--------|----------:|-----------:|------:|----------:|-------------:|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._format_currency(t.get('energy_cost', 0))} "
                f"| {self._format_currency(t.get('demand_cost', 0))} "
                f"| {self._format_currency(t.get('total_cost', 0))} "
                f"| {self._fmt(t.get('vs_budget_pct', 0))}% "
                f"| {self._fmt(t.get('vs_prior_year_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_enpi_performance(self, data: Dict[str, Any]) -> str:
        """Render EnPI performance section."""
        enpis = data.get("enpi_performance", [])
        if not enpis:
            return "## 3. EnPI Performance\n\n_No EnPI performance data available._"
        lines = [
            "## 3. EnPI Performance\n",
            "| EnPI | Baseline | Current | Improvement | Target | On Track |",
            "|------|--------:|---------:|----------:|-------:|----------|",
        ]
        for e in enpis:
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {self._fmt(e.get('baseline', 0), 3)} "
                f"| {self._fmt(e.get('current', 0), 3)} "
                f"| {self._fmt(e.get('improvement_pct', 0))}% "
                f"| {self._fmt(e.get('target', 0), 3)} "
                f"| {e.get('on_track', '-')} |"
            )
        return "\n".join(lines)

    def _md_top_anomalies(self, data: Dict[str, Any]) -> str:
        """Render top anomalies section."""
        anomalies = data.get("top_anomalies", [])
        if not anomalies:
            return "## 4. Top Anomalies\n\n_No anomalies requiring attention._"
        lines = [
            "## 4. Top Anomalies Requiring Attention\n",
            "| # | System | Description | Severity | Est. Waste (MWh/yr) | Status |",
            "|---|--------|-------------|----------|-------------------:|--------|",
        ]
        for i, a in enumerate(anomalies, 1):
            lines.append(
                f"| {i} | {a.get('system', '-')} "
                f"| {a.get('description', '-')} "
                f"| {a.get('severity', '-')} "
                f"| {self._fmt(a.get('waste_mwh_yr', 0), 1)} "
                f"| {a.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_savings_achieved(self, data: Dict[str, Any]) -> str:
        """Render savings achieved section."""
        savings = data.get("savings_achieved", {})
        if not savings:
            return "## 5. Savings Achieved\n\n_No savings data available._"
        projects = savings.get("projects", [])
        lines = [
            "## 5. Savings Achieved\n",
            f"**Total Energy Savings:** {self._format_energy(savings.get('total_energy_mwh', 0))}  \n"
            f"**Total Cost Savings:** {self._format_currency(savings.get('total_cost_savings', 0))}  \n"
            f"**Total Carbon Avoided:** {self._fmt(savings.get('total_carbon_tco2', 0), 1)} tCO2\n",
        ]
        if projects:
            lines.append("| Project | Energy (MWh) | Cost Savings | Carbon (tCO2) | Status |")
            lines.append("|---------|----------:|-----------:|-------------:|--------|")
            for p in projects:
                lines.append(
                    f"| {p.get('project', '-')} "
                    f"| {self._fmt(p.get('energy_mwh', 0), 1)} "
                    f"| {self._format_currency(p.get('cost_savings', 0))} "
                    f"| {self._fmt(p.get('carbon_tco2', 0), 1)} "
                    f"| {p.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items section."""
        actions = data.get("action_items", [])
        if not actions:
            return "## 6. Action Items\n\n_No action items._"
        lines = [
            "## 6. Action Items\n",
            "| # | Action | Owner | Priority | Due Date | Est. Savings |",
            "|---|--------|-------|----------|----------|-----------:|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('priority', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {self._format_currency(a.get('est_savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_outlook(self, data: Dict[str, Any]) -> str:
        """Render outlook section."""
        outlook = data.get("outlook", {})
        if not outlook:
            return "## 7. Outlook\n\n_No outlook data available._"
        return (
            "## 7. Outlook\n\n"
            f"**Next Period Forecast:** {self._format_currency(outlook.get('forecast_cost', 0))}  \n"
            f"**Budget Remaining:** {self._format_currency(outlook.get('budget_remaining', 0))}  \n"
            f"**Projected Year-End Variance:** {self._format_currency(outlook.get('projected_variance', 0))}  \n"
            f"**Key Risks:** {outlook.get('key_risks', '-')}  \n"
            f"**Key Opportunities:** {outlook.get('key_opportunities', '-')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-039 Energy Monitoring Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Energy Monitoring Executive Summary</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'Prepared for: {data.get("prepared_for", "Management")}</p>'
        )

    def _html_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards."""
        kpis = data.get("kpis", {})
        return (
            '<h2>Key Performance Indicators</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Consumption</span>'
            f'<span class="value">{self._fmt(kpis.get("consumption_mwh", 0), 0)} MWh</span>'
            f'<span class="label">{self._fmt(kpis.get("consumption_change_pct", 0))}% vs prev</span></div>\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">{self._format_currency(kpis.get("total_cost", 0))}</span>'
            f'<span class="label">{self._fmt(kpis.get("cost_change_pct", 0))}% vs prev</span></div>\n'
            f'  <div class="card"><span class="label">EnPI</span>'
            f'<span class="value">{self._fmt(kpis.get("enpi_value", 0), 3)}</span>'
            f'<span class="label">{self._fmt(kpis.get("enpi_change_pct", 0))}% improvement</span></div>\n'
            f'  <div class="card"><span class="label">Peak Demand</span>'
            f'<span class="value">{self._fmt(kpis.get("peak_demand_kw", 0), 0)} kW</span>'
            f'<span class="label">{self._fmt(kpis.get("peak_change_pct", 0))}% vs prev</span></div>\n'
            f'  <div class="card"><span class="label">Carbon</span>'
            f'<span class="value">{self._fmt(kpis.get("carbon_tco2", 0), 0)} tCO2</span>'
            f'<span class="label">{self._fmt(kpis.get("carbon_change_pct", 0))}% vs prev</span></div>\n'
            '</div>'
        )

    def _html_cost_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML cost trends table."""
        trends = data.get("cost_trends", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._format_currency(t.get("energy_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("demand_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("total_cost", 0))}</td>'
                f'<td>{self._fmt(t.get("vs_budget_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Cost Trends</h2>\n'
            '<table>\n<tr><th>Period</th><th>Energy Cost</th><th>Demand Cost</th>'
            f'<th>Total</th><th>vs Budget</th></tr>\n{rows}</table>'
        )

    def _html_enpi_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI performance table."""
        enpis = data.get("enpi_performance", [])
        rows = ""
        for e in enpis:
            on_track = e.get("on_track", "")
            cls = "severity-low" if on_track == "Yes" else "severity-high"
            rows += (
                f'<tr><td>{e.get("name", "-")}</td>'
                f'<td>{self._fmt(e.get("baseline", 0), 3)}</td>'
                f'<td>{self._fmt(e.get("current", 0), 3)}</td>'
                f'<td>{self._fmt(e.get("improvement_pct", 0))}%</td>'
                f'<td class="{cls}">{on_track}</td></tr>\n'
            )
        return (
            '<h2>EnPI Performance</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Baseline</th><th>Current</th>'
            f'<th>Improvement</th><th>On Track</th></tr>\n{rows}</table>'
        )

    def _html_top_anomalies(self, data: Dict[str, Any]) -> str:
        """Render HTML top anomalies table."""
        anomalies = data.get("top_anomalies", [])
        rows = ""
        for i, a in enumerate(anomalies, 1):
            sev = a.get("severity", "low").lower()
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{a.get("system", "-")}</td>'
                f'<td>{a.get("description", "-")}</td>'
                f'<td class="severity-{sev}">{a.get("severity", "-")}</td>'
                f'<td>{self._fmt(a.get("waste_mwh_yr", 0), 1)}</td>'
                f'<td>{a.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Top Anomalies</h2>\n'
            '<table>\n<tr><th>#</th><th>System</th><th>Description</th>'
            f'<th>Severity</th><th>Waste (MWh/yr)</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_savings_achieved(self, data: Dict[str, Any]) -> str:
        """Render HTML savings achieved cards and table."""
        savings = data.get("savings_achieved", {})
        projects = savings.get("projects", [])
        rows = ""
        for p in projects:
            rows += (
                f'<tr><td>{p.get("project", "-")}</td>'
                f'<td>{self._fmt(p.get("energy_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(p.get("cost_savings", 0))}</td>'
                f'<td>{p.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Savings Achieved</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Energy Saved</span>'
            f'<span class="value">{self._fmt(savings.get("total_energy_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Saved</span>'
            f'<span class="value">{self._format_currency(savings.get("total_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Carbon Avoided</span>'
            f'<span class="value">{self._fmt(savings.get("total_carbon_tco2", 0), 0)} tCO2</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Project</th><th>Energy (MWh)</th>'
            f'<th>Cost Savings</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items table."""
        actions = data.get("action_items", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            prio = a.get("priority", "").lower()
            cls = "severity-high" if prio == "high" else (
                "severity-medium" if prio == "medium" else "")
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td class="{cls}">{a.get("priority", "-")}</td>'
                f'<td>{a.get("due_date", "-")}</td>'
                f'<td>{self._format_currency(a.get("est_savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>Action Items</h2>\n'
            '<table>\n<tr><th>#</th><th>Action</th><th>Owner</th>'
            f'<th>Priority</th><th>Due Date</th><th>Est. Savings</th></tr>\n{rows}</table>'
        )

    def _html_outlook(self, data: Dict[str, Any]) -> str:
        """Render HTML outlook section."""
        outlook = data.get("outlook", {})
        proj_var = outlook.get("projected_variance", 0)
        cls = "severity-high" if proj_var > 0 else "severity-low"
        return (
            '<h2>Outlook</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Forecast Cost</span>'
            f'<span class="value">{self._format_currency(outlook.get("forecast_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Budget Remaining</span>'
            f'<span class="value">{self._format_currency(outlook.get("budget_remaining", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Projected Variance</span>'
            f'<span class="value {cls}">{self._format_currency(proj_var)}</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON KPIs."""
        kpis = data.get("kpis", {})
        return {
            "consumption_mwh": kpis.get("consumption_mwh", 0),
            "total_cost": kpis.get("total_cost", 0),
            "enpi_value": kpis.get("enpi_value", 0),
            "peak_demand_kw": kpis.get("peak_demand_kw", 0),
            "carbon_tco2": kpis.get("carbon_tco2", 0),
            "consumption_change_pct": kpis.get("consumption_change_pct", 0),
            "cost_change_pct": kpis.get("cost_change_pct", 0),
            "enpi_change_pct": kpis.get("enpi_change_pct", 0),
        }

    def _json_savings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON savings summary."""
        savings = data.get("savings_achieved", {})
        return {
            "total_energy_mwh": savings.get("total_energy_mwh", 0),
            "total_cost_savings": savings.get("total_cost_savings", 0),
            "total_carbon_tco2": savings.get("total_carbon_tco2", 0),
            "projects": savings.get("projects", []),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trends = data.get("cost_trends", [])
        enpis = data.get("enpi_performance", [])
        anomalies = data.get("top_anomalies", [])
        return {
            "cost_trend": {
                "type": "stacked_bar",
                "labels": [t.get("period", "") for t in trends],
                "series": {
                    "energy": [t.get("energy_cost", 0) for t in trends],
                    "demand": [t.get("demand_cost", 0) for t in trends],
                },
            },
            "enpi_radar": {
                "type": "radar",
                "labels": [e.get("name", "") for e in enpis],
                "series": {
                    "current": [e.get("current", 0) for e in enpis],
                    "target": [e.get("target", 0) for e in enpis],
                },
            },
            "anomaly_impact": {
                "type": "horizontal_bar",
                "labels": [a.get("system", "") for a in anomalies],
                "values": [a.get("waste_mwh_yr", 0) for a in anomalies],
            },
            "kpi_gauge": {
                "type": "gauge",
                "metrics": {
                    "consumption": data.get("kpis", {}).get("consumption_change_pct", 0),
                    "cost": data.get("kpis", {}).get("cost_change_pct", 0),
                    "enpi": data.get("kpis", {}).get("enpi_change_pct", 0),
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
            ".severity-high,.severity-critical{color:#dc3545;font-weight:700;}"
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
