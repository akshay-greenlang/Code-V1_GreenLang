# -*- coding: utf-8 -*-
"""
GapAnalysisReportTemplate - End-use gap analysis report for PACK-035.

Generates gap analysis reports that decompose the overall performance gap
between a facility and its benchmark into end-use categories (HVAC,
lighting, plug loads, etc.), identify savings potential per end-use,
prioritise improvement areas, and link to PACK-033 quick wins where
applicable.

Sections:
    1. Header
    2. Overall Performance Gap
    3. End-Use Breakdown (waterfall chart data)
    4. Gap by Category Table
    5. Savings Potential by End-Use
    6. Priority Improvement Areas
    7. Linked Quick Wins (from PACK-033)
    8. Action Plan
    9. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GapAnalysisReportTemplate:
    """
    End-use gap analysis report template.

    Renders gap analysis reports with overall performance gap summary,
    end-use waterfall decomposition, savings potential, improvement
    priorities, and linked PACK-033 quick wins across markdown,
    HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GapAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render gap analysis report as Markdown.

        Args:
            data: Gap analysis data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overall_gap(data),
            self._md_end_use_breakdown(data),
            self._md_gap_by_category(data),
            self._md_savings_potential(data),
            self._md_priority_areas(data),
            self._md_linked_quick_wins(data),
            self._md_action_plan(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render gap analysis report as self-contained HTML.

        Args:
            data: Gap analysis data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overall_gap(data),
            self._html_end_use_breakdown(data),
            self._html_gap_by_category(data),
            self._html_savings_potential(data),
            self._html_priority_areas(data),
            self._html_linked_quick_wins(data),
            self._html_action_plan(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Gap Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render gap analysis report as structured JSON.

        Args:
            data: Gap analysis data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "gap_analysis_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "overall_gap": data.get("overall_gap", {}),
            "end_use_breakdown": data.get("end_use_breakdown", []),
            "gap_by_category": data.get("gap_by_category", []),
            "savings_potential": data.get("savings_potential", []),
            "priority_areas": data.get("priority_areas", []),
            "linked_quick_wins": data.get("linked_quick_wins", []),
            "action_plan": data.get("action_plan", []),
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
            "# Energy Gap Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Benchmark Source:** {data.get('benchmark_source', '-')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 GapAnalysisReportTemplate v35.0.0\n\n---"
        )

    def _md_overall_gap(self, data: Dict[str, Any]) -> str:
        """Render overall performance gap section."""
        og = data.get("overall_gap", {})
        return (
            "## 1. Overall Performance Gap\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Facility EUI | {self._fmt(og.get('facility_eui', 0))} kWh/m2/yr |\n"
            f"| Benchmark EUI | {self._fmt(og.get('benchmark_eui', 0))} kWh/m2/yr |\n"
            f"| Absolute Gap | {self._fmt(og.get('absolute_gap', 0))} kWh/m2/yr |\n"
            f"| Relative Gap | {self._fmt(og.get('relative_gap_pct', 0))}% |\n"
            f"| Total Excess Energy | {self._fmt(og.get('excess_energy_kwh', 0), 0)} kWh/yr |\n"
            f"| Excess Energy Cost | EUR {self._fmt(og.get('excess_cost_eur', 0))} /yr |\n"
            f"| Excess CO2 Emissions | {self._fmt(og.get('excess_co2_kg', 0), 0)} kg CO2/yr |"
        )

    def _md_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render end-use breakdown (waterfall) section."""
        breakdown = data.get("end_use_breakdown", [])
        if not breakdown:
            return "## 2. End-Use Breakdown\n\n_No end-use breakdown data._"
        lines = [
            "## 2. End-Use Breakdown\n",
            "| End Use | Facility (kWh/m2) | Benchmark (kWh/m2) | Gap (kWh/m2) | Gap (%) | Cumulative Gap (%) |",
            "|---------|-------------------|-------------------|-------------|---------|-------------------|",
        ]
        for eu in breakdown:
            lines.append(
                f"| {eu.get('end_use', '-')} "
                f"| {self._fmt(eu.get('facility_kwh_m2', 0))} "
                f"| {self._fmt(eu.get('benchmark_kwh_m2', 0))} "
                f"| {self._fmt(eu.get('gap_kwh_m2', 0))} "
                f"| {self._fmt(eu.get('gap_pct', 0))}% "
                f"| {self._fmt(eu.get('cumulative_gap_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_gap_by_category(self, data: Dict[str, Any]) -> str:
        """Render gap by category table section."""
        categories = data.get("gap_by_category", [])
        if not categories:
            return "## 3. Gap by Category\n\n_No category data._"
        lines = [
            "## 3. Gap by Category\n",
            "| Category | Gap (kWh/m2) | Share of Total Gap (%) | Addressable (%) |",
            "|----------|-------------|----------------------|----------------|",
        ]
        for c in categories:
            lines.append(
                f"| {c.get('category', '-')} "
                f"| {self._fmt(c.get('gap_kwh_m2', 0))} "
                f"| {self._fmt(c.get('share_of_total_pct', 0))}% "
                f"| {self._fmt(c.get('addressable_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_savings_potential(self, data: Dict[str, Any]) -> str:
        """Render savings potential by end-use section."""
        savings = data.get("savings_potential", [])
        if not savings:
            return "## 4. Savings Potential\n\n_No savings data._"
        lines = [
            "## 4. Savings Potential by End-Use\n",
            "| End Use | Savings (kWh/yr) | Cost Savings (EUR/yr) | CO2 Savings (kg/yr) | Feasibility |",
            "|---------|-----------------|---------------------|-------------------|------------|",
        ]
        total_kwh = 0
        total_cost = 0
        total_co2 = 0
        for s in savings:
            kwh = s.get("savings_kwh_yr", 0)
            cost = s.get("cost_savings_eur", 0)
            co2 = s.get("co2_savings_kg", 0)
            total_kwh += kwh
            total_cost += cost
            total_co2 += co2
            lines.append(
                f"| {s.get('end_use', '-')} "
                f"| {self._fmt(kwh, 0)} "
                f"| {self._fmt(cost)} "
                f"| {self._fmt(co2, 0)} "
                f"| {s.get('feasibility', '-')} |"
            )
        lines.append(
            f"| **TOTAL** | **{self._fmt(total_kwh, 0)}** "
            f"| **{self._fmt(total_cost)}** "
            f"| **{self._fmt(total_co2, 0)}** | - |"
        )
        return "\n".join(lines)

    def _md_priority_areas(self, data: Dict[str, Any]) -> str:
        """Render priority improvement areas section."""
        priorities = data.get("priority_areas", [])
        if not priorities:
            return "## 5. Priority Improvement Areas\n\n_No priorities defined._"
        lines = [
            "## 5. Priority Improvement Areas\n",
            "| Priority | Area | Gap Contribution | Ease | Impact | Score |",
            "|----------|------|-----------------|------|--------|-------|",
        ]
        for p in priorities:
            lines.append(
                f"| {p.get('priority', '-')} "
                f"| {p.get('area', '-')} "
                f"| {self._fmt(p.get('gap_contribution_pct', 0))}% "
                f"| {p.get('ease', '-')} "
                f"| {p.get('impact', '-')} "
                f"| {self._fmt(p.get('score', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_linked_quick_wins(self, data: Dict[str, Any]) -> str:
        """Render linked PACK-033 quick wins section."""
        qw = data.get("linked_quick_wins", [])
        if not qw:
            return "## 6. Linked Quick Wins (PACK-033)\n\n_No linked quick wins found._"
        lines = [
            "## 6. Linked Quick Wins (PACK-033)\n",
            "| Quick Win ID | Name | End Use | Savings (kWh/yr) | Payback (months) |",
            "|-------------|------|---------|-----------------|-----------------|",
        ]
        for q in qw:
            lines.append(
                f"| {q.get('quick_win_id', '-')} "
                f"| {q.get('name', '-')} "
                f"| {q.get('end_use', '-')} "
                f"| {self._fmt(q.get('savings_kwh_yr', 0), 0)} "
                f"| {self._fmt(q.get('payback_months', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_action_plan(self, data: Dict[str, Any]) -> str:
        """Render action plan section."""
        actions = data.get("action_plan", [])
        if not actions:
            return "## 7. Action Plan\n\n_No actions defined._"
        lines = [
            "## 7. Action Plan\n",
            "| # | Action | End Use | Timeline | Owner | Cost Estimate |",
            "|---|--------|---------|----------|-------|--------------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('end_use', '-')} "
                f"| {a.get('timeline', '-')} "
                f"| {a.get('owner', '-')} "
                f"| EUR {self._fmt(a.get('cost_estimate', 0))} |"
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
            f'<h1>Energy Gap Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Benchmark: {data.get("benchmark_source", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_overall_gap(self, data: Dict[str, Any]) -> str:
        """Render HTML overall gap cards."""
        og = data.get("overall_gap", {})
        gap_pct = og.get("relative_gap_pct", 0)
        gap_cls = "card-red" if gap_pct > 0 else "card-green"
        return (
            '<h2>Overall Performance Gap</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Facility EUI</span>'
            f'<span class="value">{self._fmt(og.get("facility_eui", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card"><span class="label">Benchmark EUI</span>'
            f'<span class="value">{self._fmt(og.get("benchmark_eui", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card {gap_cls}"><span class="label">Gap</span>'
            f'<span class="value">{self._fmt(gap_pct)}%</span></div>\n'
            f'  <div class="card"><span class="label">Excess Cost</span>'
            f'<span class="value">EUR {self._fmt(og.get("excess_cost_eur", 0))}</span>'
            f'<span class="label">per year</span></div>\n'
            '</div>'
        )

    def _html_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML end-use breakdown table."""
        breakdown = data.get("end_use_breakdown", [])
        rows = ""
        for eu in breakdown:
            gap = eu.get("gap_kwh_m2", 0)
            cls = "gap-positive" if gap > 0 else "gap-negative"
            rows += (
                f'<tr><td>{eu.get("end_use", "-")}</td>'
                f'<td>{self._fmt(eu.get("facility_kwh_m2", 0))}</td>'
                f'<td>{self._fmt(eu.get("benchmark_kwh_m2", 0))}</td>'
                f'<td class="{cls}">{self._fmt(gap)}</td>'
                f'<td>{self._fmt(eu.get("gap_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>End-Use Breakdown</h2>\n'
            '<table>\n<tr><th>End Use</th><th>Facility</th><th>Benchmark</th>'
            f'<th>Gap</th><th>Gap %</th></tr>\n{rows}</table>'
        )

    def _html_gap_by_category(self, data: Dict[str, Any]) -> str:
        """Render HTML gap by category table."""
        categories = data.get("gap_by_category", [])
        rows = "".join(
            f'<tr><td>{c.get("category", "-")}</td>'
            f'<td>{self._fmt(c.get("gap_kwh_m2", 0))}</td>'
            f'<td>{self._fmt(c.get("share_of_total_pct", 0))}%</td>'
            f'<td>{self._fmt(c.get("addressable_pct", 0))}%</td></tr>\n'
            for c in categories
        )
        return (
            '<h2>Gap by Category</h2>\n'
            '<table>\n<tr><th>Category</th><th>Gap (kWh/m2)</th>'
            f'<th>Share</th><th>Addressable</th></tr>\n{rows}</table>'
        )

    def _html_savings_potential(self, data: Dict[str, Any]) -> str:
        """Render HTML savings potential table."""
        savings = data.get("savings_potential", [])
        rows = "".join(
            f'<tr><td>{s.get("end_use", "-")}</td>'
            f'<td>{self._fmt(s.get("savings_kwh_yr", 0), 0)}</td>'
            f'<td>EUR {self._fmt(s.get("cost_savings_eur", 0))}</td>'
            f'<td>{s.get("feasibility", "-")}</td></tr>\n'
            for s in savings
        )
        return (
            '<h2>Savings Potential</h2>\n'
            '<table>\n<tr><th>End Use</th><th>Savings (kWh/yr)</th>'
            f'<th>Cost Savings</th><th>Feasibility</th></tr>\n{rows}</table>'
        )

    def _html_priority_areas(self, data: Dict[str, Any]) -> str:
        """Render HTML priority improvement areas."""
        priorities = data.get("priority_areas", [])
        items = "".join(
            f'<div class="priority-item"><strong>#{p.get("priority", "-")} '
            f'{p.get("area", "-")}</strong> | '
            f'Gap: {self._fmt(p.get("gap_contribution_pct", 0))}% | '
            f'Ease: {p.get("ease", "-")} | '
            f'Impact: {p.get("impact", "-")}</div>\n'
            for p in priorities
        )
        return f'<h2>Priority Improvement Areas</h2>\n{items}'

    def _html_linked_quick_wins(self, data: Dict[str, Any]) -> str:
        """Render HTML linked PACK-033 quick wins."""
        qw = data.get("linked_quick_wins", [])
        rows = "".join(
            f'<tr><td>{q.get("quick_win_id", "-")}</td>'
            f'<td>{q.get("name", "-")}</td>'
            f'<td>{q.get("end_use", "-")}</td>'
            f'<td>{self._fmt(q.get("savings_kwh_yr", 0), 0)}</td>'
            f'<td>{self._fmt(q.get("payback_months", 0), 1)} mo</td></tr>\n'
            for q in qw
        )
        return (
            '<h2>Linked Quick Wins (PACK-033)</h2>\n'
            '<table>\n<tr><th>ID</th><th>Name</th><th>End Use</th>'
            f'<th>Savings</th><th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_action_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML action plan."""
        actions = data.get("action_plan", [])
        items = "".join(
            f'<li><strong>{a.get("action", "-")}</strong> '
            f'({a.get("end_use", "-")}) | '
            f'Timeline: {a.get("timeline", "-")} | '
            f'Owner: {a.get("owner", "-")} | '
            f'Cost: EUR {self._fmt(a.get("cost_estimate", 0))}</li>\n'
            for a in actions
        )
        return f'<h2>Action Plan</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        breakdown = data.get("end_use_breakdown", [])
        savings = data.get("savings_potential", [])
        return {
            "gap_waterfall": {
                "type": "waterfall",
                "labels": [eu.get("end_use", "") for eu in breakdown],
                "values": [eu.get("gap_kwh_m2", 0) for eu in breakdown],
            },
            "savings_bar": {
                "type": "bar",
                "labels": [s.get("end_use", "") for s in savings],
                "values": [s.get("savings_kwh_yr", 0) for s in savings],
            },
            "gap_pie": {
                "type": "pie",
                "labels": [eu.get("end_use", "") for eu in breakdown],
                "values": [abs(eu.get("gap_kwh_m2", 0)) for eu in breakdown],
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
            ".card-red{background:#f8d7da;}"
            ".card-green{background:#d1e7dd;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".gap-positive{color:#dc3545;font-weight:600;}"
            ".gap-negative{color:#198754;font-weight:600;}"
            ".priority-item{background:#fff3cd;border-left:4px solid #ffc107;"
            "padding:10px 15px;margin:8px 0;border-radius:0 4px 4px 0;}"
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
