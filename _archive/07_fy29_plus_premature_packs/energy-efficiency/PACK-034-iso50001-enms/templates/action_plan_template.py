# -*- coding: utf-8 -*-
"""
ActionPlanTemplate - ISO 50001 Clause 6.2 Objectives & Action Plans for PACK-034.

Generates comprehensive action plan documents aligned with ISO 50001:2018
Clause 6.2. Covers objectives summary, targets table, detailed action plans
per objective with timeline and resources, financial summary including
investment, savings and payback, implementation schedule with Gantt-style
data, risk assessment, and progress tracking framework.

Sections:
    1. Objectives Summary
    2. Targets Table
    3. Action Plans (per objective)
    4. Financial Summary
    5. Implementation Schedule
    6. Risk Assessment
    7. Progress Tracking Framework

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActionPlanTemplate:
    """
    ISO 50001 objectives and action plans template.

    Renders action plan documents aligned with ISO 50001:2018 Clause 6.2,
    including objectives, targets, action plans, financial analysis,
    implementation schedule, risk assessment, and progress tracking
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ActionPlanTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render action plan document as Markdown.

        Args:
            data: Action plan data including objectives, targets,
                  action_plans, financial_summary, and schedule_data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_objectives_summary(data),
            self._md_targets_table(data),
            self._md_action_plans(data),
            self._md_financial_summary(data),
            self._md_implementation_schedule(data),
            self._md_risk_assessment(data),
            self._md_progress_tracking(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render action plan document as self-contained HTML.

        Args:
            data: Action plan data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_objectives_summary(data),
            self._html_targets_table(data),
            self._html_action_plans(data),
            self._html_financial_summary(data),
            self._html_implementation_schedule(data),
            self._html_risk_assessment(data),
            self._html_progress_tracking(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Action Plan - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render action plan document as structured JSON.

        Args:
            data: Action plan data dict.

        Returns:
            Dict with structured action plan sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "action_plan",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "objectives": data.get("objectives", []),
            "targets": data.get("targets", []),
            "action_plans": data.get("action_plans", []),
            "financial_summary": self._json_financial_summary(data),
            "schedule": data.get("schedule_data", []),
            "risk_assessment": data.get("risk_assessment", []),
            "progress_tracking": data.get("progress_tracking", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with action plan metadata."""
        org = data.get("organization_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Energy Objectives & Action Plans\n\n"
            f"**Organization:** {org}  \n"
            f"**Plan Period:** {data.get('plan_period', '')}  \n"
            f"**ISO 50001:2018 Clause:** 6.2  \n"
            f"**Plan Version:** {data.get('plan_version', '1.0')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 ActionPlanTemplate v34.0.0\n\n---"
        )

    def _md_objectives_summary(self, data: Dict[str, Any]) -> str:
        """Render objectives summary section."""
        objectives = data.get("objectives", [])
        if not objectives:
            return "## 1. Objectives Summary\n\n_No objectives defined._"
        lines = [
            "## 1. Objectives Summary\n",
            "Energy objectives consistent with the energy policy (ISO 50001 Clause 6.2):\n",
            "| # | Objective | Category | Priority | Owner | Status |",
            "|---|-----------|----------|----------|-------|--------|",
        ]
        for i, obj in enumerate(objectives, 1):
            lines.append(
                f"| {i} | {obj.get('objective', '-')} "
                f"| {obj.get('category', '-')} "
                f"| {obj.get('priority', '-')} "
                f"| {obj.get('owner', '-')} "
                f"| {obj.get('status', 'Not Started')} |"
            )
        return "\n".join(lines)

    def _md_targets_table(self, data: Dict[str, Any]) -> str:
        """Render targets table section."""
        targets = data.get("targets", [])
        if not targets:
            return "## 2. Targets\n\n_No targets defined._"
        lines = [
            "## 2. Targets\n",
            "| Objective | Target | EnPI | Baseline | Target Value | Deadline | Measurement |",
            "|-----------|--------|------|----------|-------------|----------|-------------|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('objective_ref', '-')} "
                f"| {t.get('target', '-')} "
                f"| {t.get('enpi', '-')} "
                f"| {t.get('baseline_value', '-')} "
                f"| {t.get('target_value', '-')} "
                f"| {t.get('deadline', '-')} "
                f"| {t.get('measurement', '-')} |"
            )
        return "\n".join(lines)

    def _md_action_plans(self, data: Dict[str, Any]) -> str:
        """Render detailed action plans per objective."""
        plans = data.get("action_plans", [])
        if not plans:
            return "## 3. Action Plans\n\n_No action plans defined._"
        lines = ["## 3. Action Plans\n"]
        for i, plan in enumerate(plans, 1):
            lines.extend([
                f"### 3.{i} {plan.get('title', 'Action Plan')}",
                "",
                f"**Objective:** {plan.get('objective_ref', '-')}  ",
                f"**Responsible:** {plan.get('responsible', '-')}  ",
                f"**Start Date:** {plan.get('start_date', '-')}  ",
                f"**End Date:** {plan.get('end_date', '-')}  ",
                f"**Budget:** {self._format_currency(plan.get('budget', 0))}  ",
                f"**Status:** {plan.get('status', 'Not Started')}",
                "",
            ])
            actions = plan.get("actions", [])
            if actions:
                lines.extend([
                    "| Step | Action | Responsible | Due Date | Resources | Status |",
                    "|------|--------|-------------|----------|-----------|--------|",
                ])
                for j, action in enumerate(actions, 1):
                    lines.append(
                        f"| {j} | {action.get('action', '-')} "
                        f"| {action.get('responsible', '-')} "
                        f"| {action.get('due_date', '-')} "
                        f"| {action.get('resources', '-')} "
                        f"| {action.get('status', '-')} |"
                    )
            lines.append("")
        return "\n".join(lines)

    def _md_financial_summary(self, data: Dict[str, Any]) -> str:
        """Render financial summary section."""
        fin = data.get("financial_summary", {})
        lines = [
            "## 4. Financial Summary\n",
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Investment Required | {self._format_currency(fin.get('total_investment', 0))} |\n"
            f"| Annual Energy Savings | {self._format_currency(fin.get('annual_savings', 0))} |\n"
            f"| Simple Payback Period | {self._fmt(fin.get('payback_months', 0), 1)} months |\n"
            f"| Net Present Value (NPV) | {self._format_currency(fin.get('npv', 0))} |\n"
            f"| Internal Rate of Return | {self._fmt(fin.get('irr_pct', 0))}% |\n"
            f"| Return on Investment | {self._fmt(fin.get('roi_pct', 0))}% |",
        ]
        by_category = fin.get("by_category", [])
        if by_category:
            lines.extend([
                "\n### Investment by Category\n",
                "| Category | Investment | Annual Savings | Payback (mo) |",
                "|----------|-----------|---------------|-------------|",
            ])
            for cat in by_category:
                lines.append(
                    f"| {cat.get('category', '-')} "
                    f"| {self._format_currency(cat.get('investment', 0))} "
                    f"| {self._format_currency(cat.get('annual_savings', 0))} "
                    f"| {self._fmt(cat.get('payback_months', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_implementation_schedule(self, data: Dict[str, Any]) -> str:
        """Render implementation schedule with Gantt-style data."""
        schedule = data.get("schedule_data", [])
        if not schedule:
            return "## 5. Implementation Schedule\n\n_No schedule data available._"
        lines = [
            "## 5. Implementation Schedule\n",
            "| Action | Phase | Start | End | Duration (wks) | Dependencies | Milestone |",
            "|--------|-------|-------|-----|---------------|-------------|-----------|",
        ]
        for s in schedule:
            deps = ", ".join(s.get("dependencies", [])) if s.get("dependencies") else "None"
            lines.append(
                f"| {s.get('action', '-')} "
                f"| {s.get('phase', '-')} "
                f"| {s.get('start_date', '-')} "
                f"| {s.get('end_date', '-')} "
                f"| {s.get('duration_weeks', '-')} "
                f"| {deps} "
                f"| {s.get('milestone', '-')} |"
            )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        risks = data.get("risk_assessment", [])
        if not risks:
            risks = [
                {"risk": "Budget overrun", "likelihood": "Medium", "impact": "High", "mitigation": "Phased implementation, contingency budget (10%)"},
                {"risk": "Implementation delays", "likelihood": "Medium", "impact": "Medium", "mitigation": "Dedicated project manager, weekly tracking"},
                {"risk": "Lower than expected savings", "likelihood": "Low", "impact": "High", "mitigation": "Conservative estimates, M&V protocol"},
                {"risk": "Staff resistance to change", "likelihood": "Medium", "impact": "Medium", "mitigation": "Training program, change management"},
            ]
        lines = [
            "## 6. Risk Assessment\n",
            "| Risk | Likelihood | Impact | Risk Level | Mitigation Strategy |",
            "|------|-----------|--------|-----------|-------------------|",
        ]
        for r in risks:
            level = self._compute_risk_level(
                r.get("likelihood", "Medium"),
                r.get("impact", "Medium"),
            )
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {level} "
                f"| {r.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_progress_tracking(self, data: Dict[str, Any]) -> str:
        """Render progress tracking framework section."""
        tracking = data.get("progress_tracking", {})
        lines = [
            "## 7. Progress Tracking Framework\n",
            f"**Review Frequency:** {tracking.get('review_frequency', 'Monthly')}  ",
            f"**Tracking Method:** {tracking.get('tracking_method', 'KPI Dashboard + Status Reports')}  ",
            f"**Escalation Path:** {tracking.get('escalation_path', 'Energy Manager -> Energy Team -> Top Management')}\n",
            "### Key Tracking Metrics\n",
        ]
        metrics = tracking.get("metrics", [
            "Percentage of actions completed on schedule",
            "Actual vs planned energy savings",
            "Actual vs planned expenditure",
            "EnPI improvement trend",
            "Number of overdue actions",
        ])
        for m in metrics:
            lines.append(f"- {m}")
        lines.extend([
            "\n### Reporting Schedule\n",
            "| Report | Frequency | Audience | Content |",
            "|--------|-----------|----------|---------|",
        ])
        reports = tracking.get("reports", [
            {"report": "Action Status Update", "frequency": "Weekly", "audience": "Energy Team", "content": "Action status, blockers, next steps"},
            {"report": "Monthly Progress Report", "frequency": "Monthly", "audience": "Management", "content": "KPIs, savings, schedule adherence"},
            {"report": "Quarterly Review", "frequency": "Quarterly", "audience": "Top Management", "content": "Objectives progress, financials, risks"},
        ])
        for rpt in reports:
            lines.append(
                f"| {rpt.get('report', '-')} "
                f"| {rpt.get('frequency', '-')} "
                f"| {rpt.get('audience', '-')} "
                f"| {rpt.get('content', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>Energy Objectives & Action Plans</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'Plan Period: {data.get("plan_period", "-")} | '
            f'ISO 50001 Clause 6.2</p>'
        )

    def _html_objectives_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML objectives summary."""
        objectives = data.get("objectives", [])
        rows = ""
        for obj in objectives:
            status_cls = self._status_class(obj.get("status", "Not Started"))
            rows += (
                f'<tr><td>{obj.get("objective", "-")}</td>'
                f'<td>{obj.get("category", "-")}</td>'
                f'<td>{obj.get("priority", "-")}</td>'
                f'<td>{obj.get("owner", "-")}</td>'
                f'<td class="{status_cls}">{obj.get("status", "Not Started")}</td></tr>\n'
            )
        return (
            '<h2>1. Objectives Summary</h2>\n'
            '<table>\n<tr><th>Objective</th><th>Category</th>'
            f'<th>Priority</th><th>Owner</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_targets_table(self, data: Dict[str, Any]) -> str:
        """Render HTML targets table."""
        targets = data.get("targets", [])
        rows = ""
        for t in targets:
            rows += (
                f'<tr><td>{t.get("objective_ref", "-")}</td>'
                f'<td>{t.get("target", "-")}</td>'
                f'<td>{t.get("enpi", "-")}</td>'
                f'<td>{t.get("target_value", "-")}</td>'
                f'<td>{t.get("deadline", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. Targets</h2>\n'
            '<table>\n<tr><th>Objective</th><th>Target</th>'
            f'<th>EnPI</th><th>Target Value</th><th>Deadline</th></tr>\n{rows}</table>'
        )

    def _html_action_plans(self, data: Dict[str, Any]) -> str:
        """Render HTML action plans."""
        plans = data.get("action_plans", [])
        html_parts = ['<h2>3. Action Plans</h2>\n']
        for plan in plans:
            actions = plan.get("actions", [])
            rows = ""
            for a in actions:
                rows += (
                    f'<tr><td>{a.get("action", "-")}</td>'
                    f'<td>{a.get("responsible", "-")}</td>'
                    f'<td>{a.get("due_date", "-")}</td>'
                    f'<td>{a.get("status", "-")}</td></tr>\n'
                )
            html_parts.append(
                f'<h3>{plan.get("title", "Action Plan")}</h3>\n'
                f'<p><strong>Responsible:</strong> {plan.get("responsible", "-")} | '
                f'<strong>Budget:</strong> {self._format_currency(plan.get("budget", 0))}</p>\n'
                f'<table>\n<tr><th>Action</th><th>Responsible</th>'
                f'<th>Due Date</th><th>Status</th></tr>\n{rows}</table>\n'
            )
        return "".join(html_parts)

    def _html_financial_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML financial summary cards."""
        fin = data.get("financial_summary", {})
        return (
            '<h2>4. Financial Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Investment</span>'
            f'<span class="value">{self._format_currency(fin.get("total_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._format_currency(fin.get("annual_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(fin.get("payback_months", 0), 1)} mo</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(fin.get("npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(fin.get("irr_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_implementation_schedule(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation schedule."""
        schedule = data.get("schedule_data", [])
        rows = ""
        for s in schedule:
            rows += (
                f'<tr><td>{s.get("action", "-")}</td>'
                f'<td>{s.get("phase", "-")}</td>'
                f'<td>{s.get("start_date", "-")}</td>'
                f'<td>{s.get("end_date", "-")}</td>'
                f'<td>{s.get("duration_weeks", "-")}</td></tr>\n'
            )
        return (
            '<h2>5. Implementation Schedule</h2>\n'
            '<table>\n<tr><th>Action</th><th>Phase</th>'
            f'<th>Start</th><th>End</th><th>Duration (wks)</th></tr>\n{rows}</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment."""
        risks = data.get("risk_assessment", [])
        rows = ""
        for r in risks:
            level = self._compute_risk_level(
                r.get("likelihood", "Medium"),
                r.get("impact", "Medium"),
            )
            cls = f"risk-{level.lower()}"
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td>{r.get("impact", "-")}</td>'
                f'<td class="{cls}">{level}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Risk Assessment</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Level</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_progress_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML progress tracking framework."""
        tracking = data.get("progress_tracking", {})
        return (
            '<h2>7. Progress Tracking Framework</h2>\n'
            f'<p><strong>Review Frequency:</strong> {tracking.get("review_frequency", "Monthly")}</p>\n'
            f'<p><strong>Tracking Method:</strong> {tracking.get("tracking_method", "KPI Dashboard")}</p>\n'
            '<h3>Key Metrics</h3>\n'
            '<ul>\n'
            '<li>Actions completed on schedule (%)</li>\n'
            '<li>Actual vs planned energy savings</li>\n'
            '<li>Actual vs planned expenditure</li>\n'
            '<li>EnPI improvement trend</li>\n'
            '</ul>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_financial_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON financial summary."""
        fin = data.get("financial_summary", {})
        return {
            "total_investment": fin.get("total_investment", 0),
            "annual_savings": fin.get("annual_savings", 0),
            "payback_months": fin.get("payback_months", 0),
            "npv": fin.get("npv", 0),
            "irr_pct": fin.get("irr_pct", 0),
            "roi_pct": fin.get("roi_pct", 0),
            "by_category": fin.get("by_category", []),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        plans = data.get("action_plans", [])
        schedule = data.get("schedule_data", [])
        by_cat = data.get("financial_summary", {}).get("by_category", [])
        return {
            "investment_pie": {
                "type": "pie",
                "labels": [c.get("category", "") for c in by_cat],
                "values": [c.get("investment", 0) for c in by_cat],
            },
            "gantt": {
                "type": "gantt",
                "tasks": [
                    {
                        "name": s.get("action", ""),
                        "start": s.get("start_date", ""),
                        "end": s.get("end_date", ""),
                        "phase": s.get("phase", ""),
                    }
                    for s in schedule
                ],
            },
            "status_donut": {
                "type": "donut",
                "labels": list({p.get("status", "") for p in plans}),
                "values": [
                    sum(1 for p in plans if p.get("status") == status)
                    for status in {p.get("status", "") for p in plans}
                ],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_risk_level(self, likelihood: str, impact: str) -> str:
        """Compute risk level from likelihood and impact."""
        risk_matrix = {
            ("High", "High"): "Critical",
            ("High", "Medium"): "High",
            ("High", "Low"): "Medium",
            ("Medium", "High"): "High",
            ("Medium", "Medium"): "Medium",
            ("Medium", "Low"): "Low",
            ("Low", "High"): "Medium",
            ("Low", "Medium"): "Low",
            ("Low", "Low"): "Low",
        }
        return risk_matrix.get((likelihood, impact), "Medium")

    def _status_class(self, status: str) -> str:
        """Return CSS class for status value."""
        status_lower = status.lower()
        if status_lower in ("completed", "complete", "done"):
            return "status-complete"
        if status_lower in ("in progress", "ongoing"):
            return "status-progress"
        if status_lower in ("overdue", "delayed"):
            return "status-overdue"
        return ""

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "h3{color:#495057;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-complete{color:#198754;font-weight:600;}"
            ".status-progress{color:#0d6efd;font-weight:600;}"
            ".status-overdue{color:#dc3545;font-weight:600;}"
            ".risk-critical{color:#dc3545;font-weight:700;}"
            ".risk-high{color:#fd7e14;font-weight:700;}"
            ".risk-medium{color:#ffc107;font-weight:600;}"
            ".risk-low{color:#198754;font-weight:500;}"
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
