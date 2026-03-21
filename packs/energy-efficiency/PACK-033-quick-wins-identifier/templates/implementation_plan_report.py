# -*- coding: utf-8 -*-
"""
ImplementationPlanReportTemplate - Implementation roadmap for PACK-033.

Generates implementation plan reports for quick-win energy efficiency
measures, including phase timelines, resource requirements, budget
breakdowns, rebate opportunities, risk mitigation strategies, and
milestone tracking.

Sections:
    1. Plan Overview
    2. Phase Timeline (Gantt-style data)
    3. Resource Requirements
    4. Budget Breakdown
    5. Rebate Opportunities
    6. Risk Mitigation
    7. Milestones

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ImplementationPlanReportTemplate:
    """
    Implementation plan report template.

    Renders phased implementation roadmaps for quick-win measures
    with resource planning, budget analysis, rebate opportunities,
    risk strategies, and milestones across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImplementationPlanReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render implementation plan report as Markdown.

        Args:
            data: Implementation planning engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_plan_overview(data),
            self._md_phase_timeline(data),
            self._md_resource_requirements(data),
            self._md_budget_breakdown(data),
            self._md_rebate_opportunities(data),
            self._md_risk_mitigation(data),
            self._md_milestones(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render implementation plan report as self-contained HTML.

        Args:
            data: Implementation planning engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_plan_overview(data),
            self._html_phase_timeline(data),
            self._html_budget_breakdown(data),
            self._html_rebate_opportunities(data),
            self._html_risk_mitigation(data),
            self._html_milestones(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Implementation Plan Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render implementation plan report as structured JSON.

        Args:
            data: Implementation planning engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "implementation_plan_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "plan_overview": self._json_plan_overview(data),
            "phases": data.get("phases", []),
            "resource_requirements": data.get("resource_requirements", []),
            "budget_breakdown": data.get("budget_breakdown", {}),
            "rebate_opportunities": data.get("rebate_opportunities", []),
            "risks": data.get("risks", []),
            "milestones": data.get("milestones", []),
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
            f"# Implementation Plan Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Plan Version:** {data.get('plan_version', '1.0')}  \n"
            f"**Plan Horizon:** {data.get('plan_horizon', '12 months')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 ImplementationPlanReportTemplate v33.0.0\n\n---"
        )

    def _md_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render plan overview section."""
        overview = data.get("plan_overview", {})
        return (
            "## 1. Plan Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Measures | {overview.get('total_measures', 0)} |\n"
            f"| Implementation Phases | {overview.get('total_phases', 0)} |\n"
            f"| Total Budget | {self._format_currency(overview.get('total_budget', 0))} |\n"
            f"| Expected Annual Savings | {self._format_currency(overview.get('expected_annual_savings', 0))} /yr |\n"
            f"| Available Rebates | {self._format_currency(overview.get('total_rebates', 0))} |\n"
            f"| Net Investment | {self._format_currency(overview.get('net_investment', 0))} |\n"
            f"| Start Date | {overview.get('start_date', '-')} |\n"
            f"| Target Completion | {overview.get('target_completion', '-')} |"
        )

    def _md_phase_timeline(self, data: Dict[str, Any]) -> str:
        """Render phase timeline section with Gantt-style data."""
        phases = data.get("phases", [])
        if not phases:
            return "## 2. Phase Timeline\n\n_No phase data available._"
        lines = ["## 2. Phase Timeline\n"]
        for phase in phases:
            lines.extend([
                f"### {phase.get('phase_name', 'Phase')}",
                f"- **Period:** {phase.get('start_date', '-')} to {phase.get('end_date', '-')}",
                f"- **Duration:** {phase.get('duration_weeks', 0)} weeks",
                f"- **Measures:** {phase.get('measure_count', 0)}",
                f"- **Budget:** {self._format_currency(phase.get('budget', 0))}",
                f"- **Expected Savings:** {self._format_currency(phase.get('expected_savings', 0))} /yr",
                "",
            ])
            measures = phase.get("measures", [])
            if measures:
                lines.append("| Measure | Start | End | Owner | Status |")
                lines.append("|---------|-------|-----|-------|--------|")
                for m in measures:
                    lines.append(
                        f"| {m.get('name', '-')} "
                        f"| {m.get('start_date', '-')} "
                        f"| {m.get('end_date', '-')} "
                        f"| {m.get('owner', '-')} "
                        f"| {m.get('status', 'Planned')} |"
                    )
            lines.append("")
        return "\n".join(lines)

    def _md_resource_requirements(self, data: Dict[str, Any]) -> str:
        """Render resource requirements section."""
        resources = data.get("resource_requirements", [])
        if not resources:
            return "## 3. Resource Requirements\n\n_No resource data available._"
        lines = [
            "## 3. Resource Requirements\n",
            "| Resource Type | Quantity | Duration | Cost | Source |",
            "|--------------|----------|----------|------|--------|",
        ]
        for r in resources:
            lines.append(
                f"| {r.get('type', '-')} "
                f"| {r.get('quantity', '-')} "
                f"| {r.get('duration', '-')} "
                f"| {self._format_currency(r.get('cost', 0))} "
                f"| {r.get('source', '-')} |"
            )
        return "\n".join(lines)

    def _md_budget_breakdown(self, data: Dict[str, Any]) -> str:
        """Render budget breakdown section."""
        budget = data.get("budget_breakdown", {})
        categories = budget.get("categories", [])
        if not categories:
            return "## 4. Budget Breakdown\n\n_No budget data available._"
        lines = [
            "## 4. Budget Breakdown\n",
            f"**Total Budget:** {self._format_currency(budget.get('total_budget', 0))}  ",
            f"**Contingency:** {self._format_currency(budget.get('contingency', 0))} "
            f"({self._fmt(budget.get('contingency_pct', 10))}%)\n",
            "| Category | Amount | Share (%) |",
            "|----------|--------|-----------|",
        ]
        for cat in categories:
            lines.append(
                f"| {cat.get('category', '-')} "
                f"| {self._format_currency(cat.get('amount', 0))} "
                f"| {self._fmt(cat.get('share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_rebate_opportunities(self, data: Dict[str, Any]) -> str:
        """Render rebate opportunities section."""
        rebates = data.get("rebate_opportunities", [])
        if not rebates:
            return "## 5. Rebate Opportunities\n\n_No rebates identified._"
        lines = [
            "## 5. Rebate Opportunities\n",
            "| Measure | Program | Rebate Amount | Application Deadline |",
            "|---------|---------|---------------|---------------------|",
        ]
        for r in rebates:
            lines.append(
                f"| {r.get('measure', '-')} "
                f"| {r.get('program', '-')} "
                f"| {self._format_currency(r.get('rebate_amount', 0))} "
                f"| {r.get('deadline', '-')} |"
            )
        total = sum(r.get("rebate_amount", 0) for r in rebates)
        lines.append(f"| **TOTAL** | | **{self._format_currency(total)}** | |")
        return "\n".join(lines)

    def _md_risk_mitigation(self, data: Dict[str, Any]) -> str:
        """Render risk mitigation section."""
        risks = data.get("risks", [])
        if not risks:
            return "## 6. Risk Mitigation\n\n_No risks identified._"
        lines = [
            "## 6. Risk Mitigation\n",
            "| Risk | Likelihood | Impact | Mitigation Strategy |",
            "|------|-----------|--------|---------------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        """Render milestones section."""
        milestones = data.get("milestones", [])
        if not milestones:
            return "## 7. Milestones\n\n_No milestones defined._"
        lines = [
            "## 7. Milestones\n",
            "| # | Milestone | Target Date | Owner | Status |",
            "|---|-----------|------------|-------|--------|",
        ]
        for i, ms in enumerate(milestones, 1):
            lines.append(
                f"| {i} | {ms.get('milestone', '-')} "
                f"| {ms.get('target_date', '-')} "
                f"| {ms.get('owner', '-')} "
                f"| {ms.get('status', 'Pending')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-033 Quick Wins Identifier Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Implementation Plan Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Horizon: {data.get("plan_horizon", "12 months")}</p>'
        )

    def _html_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML plan overview cards."""
        o = data.get("plan_overview", {})
        return (
            '<h2>Plan Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Measures</span>'
            f'<span class="value">{o.get("total_measures", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Total Budget</span>'
            f'<span class="value">{self._format_currency(o.get("total_budget", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._format_currency(o.get("expected_annual_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Net Investment</span>'
            f'<span class="value">{self._format_currency(o.get("net_investment", 0))}</span></div>\n'
            '</div>'
        )

    def _html_phase_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML phase timeline."""
        phases = data.get("phases", [])
        content = ""
        for phase in phases:
            actions = "".join(
                f'<li>{m.get("name", "-")} (Owner: {m.get("owner", "-")})</li>'
                for m in phase.get("measures", [])
            )
            content += (
                f'<div class="phase"><h3>{phase.get("phase_name", "Phase")} '
                f'({phase.get("start_date", "-")} to {phase.get("end_date", "-")})</h3>'
                f'<p>Budget: {self._format_currency(phase.get("budget", 0))} | '
                f'Measures: {phase.get("measure_count", 0)}</p>'
                f'<ul>{actions}</ul></div>\n'
            )
        return f'<h2>Phase Timeline</h2>\n{content}'

    def _html_budget_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML budget breakdown."""
        categories = data.get("budget_breakdown", {}).get("categories", [])
        rows = ""
        for cat in categories:
            rows += (
                f'<tr><td>{cat.get("category", "-")}</td>'
                f'<td>{self._format_currency(cat.get("amount", 0))}</td>'
                f'<td>{self._fmt(cat.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Budget Breakdown</h2>\n'
            '<table>\n<tr><th>Category</th><th>Amount</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_rebate_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML rebate opportunities."""
        rebates = data.get("rebate_opportunities", [])
        rows = ""
        for r in rebates:
            rows += (
                f'<tr><td>{r.get("measure", "-")}</td>'
                f'<td>{r.get("program", "-")}</td>'
                f'<td>{self._format_currency(r.get("rebate_amount", 0))}</td></tr>\n'
            )
        return (
            '<h2>Rebate Opportunities</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Program</th>'
            f'<th>Rebate</th></tr>\n{rows}</table>'
        )

    def _html_risk_mitigation(self, data: Dict[str, Any]) -> str:
        """Render HTML risk mitigation."""
        risks = data.get("risks", [])
        rows = ""
        for r in risks:
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td>{r.get("impact", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Risk Mitigation</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        """Render HTML milestones."""
        milestones = data.get("milestones", [])
        rows = ""
        for ms in milestones:
            status = ms.get("status", "Pending")
            cls = "status-pass" if status == "Complete" else ""
            rows += (
                f'<tr><td>{ms.get("milestone", "-")}</td>'
                f'<td>{ms.get("target_date", "-")}</td>'
                f'<td>{ms.get("owner", "-")}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            '<h2>Milestones</h2>\n'
            '<table>\n<tr><th>Milestone</th><th>Target Date</th>'
            f'<th>Owner</th><th>Status</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_plan_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON plan overview."""
        o = data.get("plan_overview", {})
        return {
            "total_measures": o.get("total_measures", 0),
            "total_phases": o.get("total_phases", 0),
            "total_budget": o.get("total_budget", 0),
            "expected_annual_savings": o.get("expected_annual_savings", 0),
            "total_rebates": o.get("total_rebates", 0),
            "net_investment": o.get("net_investment", 0),
            "start_date": o.get("start_date", ""),
            "target_completion": o.get("target_completion", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        phases = data.get("phases", [])
        categories = data.get("budget_breakdown", {}).get("categories", [])
        milestones = data.get("milestones", [])
        return {
            "gantt": {
                "type": "gantt",
                "phases": [
                    {
                        "name": p.get("phase_name", ""),
                        "start": p.get("start_date", ""),
                        "end": p.get("end_date", ""),
                        "measures": len(p.get("measures", [])),
                    }
                    for p in phases
                ],
            },
            "budget_pie": {
                "type": "pie",
                "labels": [c.get("category", "") for c in categories],
                "values": [c.get("amount", 0) for c in categories],
            },
            "milestone_timeline": {
                "type": "timeline",
                "items": [
                    {
                        "label": ms.get("milestone", ""),
                        "date": ms.get("target_date", ""),
                        "status": ms.get("status", "Pending"),
                    }
                    for ms in milestones
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
            "h3{color:#0d6efd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".phase{border-left:3px solid #0d6efd;padding-left:15px;margin:15px 0;}"
            ".status-pass{color:#198754;font-weight:600;}"
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
