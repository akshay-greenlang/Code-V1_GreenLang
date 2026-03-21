# -*- coding: utf-8 -*-
"""
PrioritizedActionsReportTemplate - Ranked actions with MCDA scores for PACK-033.

Generates a prioritized actions report with Multi-Criteria Decision Analysis
(MCDA) scoring, Pareto frontier analysis, implementation phases, and
dependency mapping for quick-win energy efficiency measures.

Sections:
    1. Ranking Summary
    2. Top 10 Actions
    3. MCDA Score Breakdown
    4. Pareto Frontier Analysis
    5. Implementation Phases
    6. Dependencies

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PrioritizedActionsReportTemplate:
    """
    Prioritized actions report template with MCDA scoring.

    Renders ranked quick-win actions with multi-criteria scores,
    Pareto analysis, phased implementation plans, and dependency
    mappings across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PrioritizedActionsReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render prioritized actions report as Markdown.

        Args:
            data: Prioritization engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_ranking_summary(data),
            self._md_top_actions(data),
            self._md_mcda_breakdown(data),
            self._md_pareto_frontier(data),
            self._md_implementation_phases(data),
            self._md_dependencies(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render prioritized actions report as self-contained HTML.

        Args:
            data: Prioritization engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_ranking_summary(data),
            self._html_top_actions(data),
            self._html_mcda_breakdown(data),
            self._html_pareto_frontier(data),
            self._html_implementation_phases(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Prioritized Actions Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render prioritized actions report as structured JSON.

        Args:
            data: Prioritization engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "prioritized_actions_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "ranking_summary": self._json_ranking_summary(data),
            "top_actions": data.get("top_actions", data.get("actions", []))[:10],
            "mcda_breakdown": data.get("mcda_breakdown", {}),
            "pareto_frontier": data.get("pareto_frontier", {}),
            "implementation_phases": data.get("implementation_phases", []),
            "dependencies": data.get("dependencies", []),
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
            f"# Prioritized Actions Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Date:** {data.get('analysis_date', '')}  \n"
            f"**MCDA Method:** {data.get('mcda_method', 'Weighted Sum Model')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 PrioritizedActionsReportTemplate v33.0.0\n\n---"
        )

    def _md_ranking_summary(self, data: Dict[str, Any]) -> str:
        """Render ranking summary section."""
        summary = data.get("ranking_summary", {})
        return (
            "## 1. Ranking Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Actions Evaluated | {summary.get('total_actions', 0)} |\n"
            f"| Actions on Pareto Frontier | {summary.get('pareto_optimal_count', 0)} |\n"
            f"| Total Potential Savings | {self._format_currency(summary.get('total_savings', 0))} /yr |\n"
            f"| Total Investment Required | {self._format_currency(summary.get('total_investment', 0))} |\n"
            f"| Weighted Average Payback | {self._fmt(summary.get('avg_payback_months', 0), 1)} months |\n"
            f"| Top-10 Savings Concentration | {self._fmt(summary.get('top10_savings_pct', 0))}% |"
        )

    def _md_top_actions(self, data: Dict[str, Any]) -> str:
        """Render top 10 actions table."""
        actions = data.get("top_actions", data.get("actions", []))[:10]
        if not actions:
            return "## 2. Top 10 Actions\n\n_No actions ranked._"
        lines = [
            "## 2. Top 10 Actions\n",
            "| Rank | Action | MCDA Score | Annual Savings | Investment | Payback (mo) |",
            "|------|--------|-----------|---------------|------------|-------------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {self._fmt(a.get('mcda_score', 0), 3)} "
                f"| {self._format_currency(a.get('annual_savings', 0))} "
                f"| {self._format_currency(a.get('investment', 0))} "
                f"| {self._fmt(a.get('payback_months', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_mcda_breakdown(self, data: Dict[str, Any]) -> str:
        """Render MCDA score breakdown section."""
        mcda = data.get("mcda_breakdown", {})
        criteria = mcda.get("criteria", [])
        if not criteria:
            return "## 3. MCDA Score Breakdown\n\n_No MCDA criteria data available._"
        lines = [
            "## 3. MCDA Score Breakdown\n",
            f"**Method:** {mcda.get('method', 'Weighted Sum Model')}  ",
            f"**Normalization:** {mcda.get('normalization', 'Min-Max')}  \n",
            "### Criteria Weights\n",
            "| Criterion | Weight | Direction | Description |",
            "|-----------|--------|-----------|-------------|",
        ]
        for c in criteria:
            lines.append(
                f"| {c.get('name', '-')} "
                f"| {self._fmt(c.get('weight', 0), 3)} "
                f"| {c.get('direction', 'maximize')} "
                f"| {c.get('description', '-')} |"
            )
        return "\n".join(lines)

    def _md_pareto_frontier(self, data: Dict[str, Any]) -> str:
        """Render Pareto frontier analysis section."""
        pareto = data.get("pareto_frontier", {})
        optimal = pareto.get("optimal_actions", [])
        if not optimal:
            return "## 4. Pareto Frontier Analysis\n\n_No Pareto analysis available._"
        lines = [
            "## 4. Pareto Frontier Analysis\n",
            f"**Objectives:** {', '.join(pareto.get('objectives', ['Cost Savings', 'Investment']))}  ",
            f"**Pareto-Optimal Actions:** {len(optimal)}\n",
            "| Action | Savings | Investment | Dominated By |",
            "|--------|---------|------------|-------------|",
        ]
        for a in optimal:
            lines.append(
                f"| {a.get('action', '-')} "
                f"| {self._format_currency(a.get('savings', 0))} "
                f"| {self._format_currency(a.get('investment', 0))} "
                f"| {a.get('dominated_by', 'None')} |"
            )
        return "\n".join(lines)

    def _md_implementation_phases(self, data: Dict[str, Any]) -> str:
        """Render implementation phases section."""
        phases = data.get("implementation_phases", [])
        if not phases:
            return "## 5. Implementation Phases\n\n_No phasing data available._"
        lines = ["## 5. Implementation Phases\n"]
        for phase in phases:
            lines.extend([
                f"### {phase.get('phase_name', 'Phase')} ({phase.get('timeframe', '-')})\n",
                f"- **Actions:** {phase.get('action_count', 0)}",
                f"- **Total Investment:** {self._format_currency(phase.get('investment', 0))}",
                f"- **Expected Savings:** {self._format_currency(phase.get('expected_savings', 0))} /yr",
                "",
            ])
            actions = phase.get("actions", [])
            for a in actions:
                lines.append(f"  - {a.get('action', '-')} (Priority: {a.get('priority', '-')})")
            lines.append("")
        return "\n".join(lines)

    def _md_dependencies(self, data: Dict[str, Any]) -> str:
        """Render action dependencies section."""
        deps = data.get("dependencies", [])
        if not deps:
            return "## 6. Dependencies\n\n_No inter-action dependencies identified._"
        lines = [
            "## 6. Dependencies\n",
            "| Action | Depends On | Type | Impact |",
            "|--------|-----------|------|--------|",
        ]
        for d in deps:
            lines.append(
                f"| {d.get('action', '-')} "
                f"| {d.get('depends_on', '-')} "
                f"| {d.get('type', '-')} "
                f"| {d.get('impact', '-')} |"
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
            f'<h1>Prioritized Actions Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Method: {data.get("mcda_method", "Weighted Sum Model")}</p>'
        )

    def _html_ranking_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML ranking summary cards."""
        s = data.get("ranking_summary", {})
        return (
            '<h2>Ranking Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Actions</span>'
            f'<span class="value">{s.get("total_actions", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Total Savings</span>'
            f'<span class="value">{self._format_currency(s.get("total_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Pareto Optimal</span>'
            f'<span class="value">{s.get("pareto_optimal_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Avg Payback</span>'
            f'<span class="value">{self._fmt(s.get("avg_payback_months", 0), 1)} mo</span></div>\n'
            '</div>'
        )

    def _html_top_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML top actions table."""
        actions = data.get("top_actions", data.get("actions", []))[:10]
        rows = ""
        for i, a in enumerate(actions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("action", "-")}</td>'
                f'<td>{self._fmt(a.get("mcda_score", 0), 3)}</td>'
                f'<td>{self._format_currency(a.get("annual_savings", 0))}</td>'
                f'<td>{self._fmt(a.get("payback_months", 0), 1)} mo</td></tr>\n'
            )
        return (
            '<h2>Top 10 Actions</h2>\n'
            '<table>\n<tr><th>#</th><th>Action</th><th>MCDA Score</th>'
            f'<th>Savings</th><th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_mcda_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML MCDA breakdown."""
        criteria = data.get("mcda_breakdown", {}).get("criteria", [])
        rows = ""
        for c in criteria:
            pct = c.get("weight", 0) * 100
            rows += (
                f'<tr><td>{c.get("name", "-")}</td>'
                f'<td>{self._fmt(c.get("weight", 0), 3)}</td>'
                f'<td><div class="bar" style="width:{pct}%"></div></td></tr>\n'
            )
        return (
            '<h2>MCDA Criteria Weights</h2>\n'
            '<table>\n<tr><th>Criterion</th><th>Weight</th>'
            f'<th>Relative Weight</th></tr>\n{rows}</table>'
        )

    def _html_pareto_frontier(self, data: Dict[str, Any]) -> str:
        """Render HTML Pareto frontier."""
        optimal = data.get("pareto_frontier", {}).get("optimal_actions", [])
        items = ""
        for a in optimal:
            items += (
                f'<div class="pareto-item"><strong>{a.get("action", "-")}</strong> | '
                f'Savings: {self._format_currency(a.get("savings", 0))} | '
                f'Investment: {self._format_currency(a.get("investment", 0))}</div>\n'
            )
        return f'<h2>Pareto Frontier</h2>\n{items}'

    def _html_implementation_phases(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation phases."""
        phases = data.get("implementation_phases", [])
        content = ""
        for phase in phases:
            actions = "".join(
                f'<li>{a.get("action", "-")}</li>'
                for a in phase.get("actions", [])
            )
            content += (
                f'<div class="phase"><h3>{phase.get("phase_name", "Phase")} - '
                f'{phase.get("timeframe", "-")}</h3>'
                f'<p>Investment: {self._format_currency(phase.get("investment", 0))} | '
                f'Savings: {self._format_currency(phase.get("expected_savings", 0))}/yr</p>'
                f'<ul>{actions}</ul></div>\n'
            )
        return f'<h2>Implementation Phases</h2>\n{content}'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_ranking_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON ranking summary."""
        s = data.get("ranking_summary", {})
        return {
            "total_actions": s.get("total_actions", 0),
            "pareto_optimal_count": s.get("pareto_optimal_count", 0),
            "total_savings": s.get("total_savings", 0),
            "total_investment": s.get("total_investment", 0),
            "avg_payback_months": s.get("avg_payback_months", 0),
            "top10_savings_pct": s.get("top10_savings_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        actions = data.get("top_actions", data.get("actions", []))[:10]
        return {
            "mcda_bar": {
                "type": "horizontal_bar",
                "labels": [a.get("action", "") for a in actions],
                "values": [a.get("mcda_score", 0) for a in actions],
            },
            "pareto_scatter": {
                "type": "scatter",
                "x_label": "Investment",
                "y_label": "Annual Savings",
                "points": [
                    {
                        "x": a.get("investment", 0),
                        "y": a.get("annual_savings", 0),
                        "label": a.get("action", ""),
                        "pareto_optimal": a.get("pareto_optimal", False),
                    }
                    for a in data.get("actions", [])
                ],
            },
            "phase_timeline": {
                "type": "gantt",
                "phases": [
                    {
                        "name": p.get("phase_name", ""),
                        "start": p.get("start_date", ""),
                        "end": p.get("end_date", ""),
                        "action_count": p.get("action_count", 0),
                    }
                    for p in data.get("implementation_phases", [])
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
            ".bar{background:#0d6efd;height:18px;border-radius:3px;}"
            ".phase{border-left:3px solid #0d6efd;padding-left:15px;margin:15px 0;}"
            ".pareto-item{background:#d1e7dd;padding:10px;margin:5px 0;border-radius:4px;}"
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
