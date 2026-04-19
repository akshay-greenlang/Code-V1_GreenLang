# -*- coding: utf-8 -*-
"""
ISO50001ReviewReportTemplate - ISO 50001 management review package for PACK-031.

Generates management review packages per ISO 50001:2018 Clause 9.3
requirements, including EnMS performance summary, EnPI trends, objectives
and targets status, internal audit results, nonconformity tracking,
energy policy review, and continual improvement evidence.

Sections:
    1. Executive Summary
    2. EnMS Performance Summary
    3. EnPI Trends & Analysis
    4. Objectives & Targets Status
    5. Internal Audit Results
    6. Nonconformities & Corrective Actions
    7. Energy Policy Review
    8. Continual Improvement Evidence
    9. Management Decisions Required

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ISO50001ReviewReportTemplate:
    """
    ISO 50001 management review package template.

    Renders management review data per ISO 50001:2018 Clause 9.3
    including EnMS performance, EnPI tracking, objectives status,
    audit results, and improvement evidence across markdown, HTML,
    and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    CLAUSE_93_INPUTS: List[str] = [
        "a) Status of actions from previous reviews",
        "b) Changes in external/internal issues",
        "c) Energy performance and EnPI improvement",
        "d) Compliance obligations",
        "e) Extent to which objectives/targets met",
        "f) Audit results",
        "g) Status of nonconformities/corrective actions",
        "h) Projected energy performance",
        "i) Recommendations for improvement",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ISO50001ReviewReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ISO 50001 management review package as Markdown.

        Args:
            data: Management review engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_previous_actions(data),
            self._md_enms_performance(data),
            self._md_enpi_trends(data),
            self._md_objectives_targets(data),
            self._md_internal_audit(data),
            self._md_nonconformities(data),
            self._md_energy_policy(data),
            self._md_continual_improvement(data),
            self._md_management_decisions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ISO 50001 management review package as HTML.

        Args:
            data: Management review engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_enpi_trends(data),
            self._html_objectives_targets(data),
            self._html_audit_results(data),
            self._html_nonconformities(data),
            self._html_improvement_evidence(data),
            self._html_management_decisions(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ISO 50001 Management Review Package</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ISO 50001 management review package as structured JSON.

        Args:
            data: Management review engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "iso_50001_review_report",
            "version": "31.0.0",
            "standard": "ISO 50001:2018",
            "clause": "9.3 Management Review",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "previous_actions": data.get("previous_actions", []),
            "enms_performance": data.get("enms_performance", {}),
            "enpi_trends": data.get("enpi_trends", {}),
            "objectives_targets": data.get("objectives_targets", []),
            "internal_audit": data.get("internal_audit", {}),
            "nonconformities": data.get("nonconformities", []),
            "energy_policy": data.get("energy_policy", {}),
            "continual_improvement": data.get("continual_improvement", {}),
            "management_decisions": data.get("management_decisions", []),
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
        org = data.get("organization_name", "Organization")
        review_date = data.get("review_date", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# ISO 50001 Management Review Package\n\n"
            f"**Organization:** {org}  \n"
            f"**Standard:** ISO 50001:2018 (Clause 9.3)  \n"
            f"**Review Date:** {review_date}  \n"
            f"**Review Period:** {data.get('review_period', '-')}  \n"
            f"**Prepared By:** {data.get('prepared_by', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 ISO50001ReviewReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Overall EnMS Status | {s.get('enms_status', '-')} |\n"
            f"| Energy Performance Improvement | {self._fmt(s.get('improvement_pct', 0))}% |\n"
            f"| EnPI Target Achievement | {self._fmt(s.get('enpi_achievement_pct', 0))}% |\n"
            f"| Objectives Met | {s.get('objectives_met', 0)} of {s.get('total_objectives', 0)} |\n"
            f"| Open Nonconformities | {s.get('open_nonconformities', 0)} |\n"
            f"| Audit Findings Closed | {s.get('findings_closed', 0)} "
            f"of {s.get('total_findings', 0)} |\n"
            f"| Energy Cost Savings | EUR {self._fmt(s.get('cost_savings_eur', 0))} |\n"
            f"| CO2 Reduction | {self._fmt(s.get('co2_reduction_tonnes', 0))} tonnes |"
        )

    def _md_previous_actions(self, data: Dict[str, Any]) -> str:
        """Render status of actions from previous reviews (Clause 9.3a)."""
        actions = data.get("previous_actions", [])
        if not actions:
            return (
                "## 2. Actions from Previous Review (9.3a)\n\n"
                "_No actions from previous review._"
            )
        lines = [
            "## 2. Actions from Previous Review (9.3a)\n",
            "| # | Action | Owner | Due Date | Status | Notes |",
            "|---|--------|-------|----------|--------|-------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {a.get('status', '-')} "
                f"| {a.get('notes', '-')} |"
            )
        completed = sum(1 for a in actions if a.get("status") == "completed")
        lines.append(f"\n**Completion Rate:** {completed}/{len(actions)} "
                     f"({self._pct(completed, len(actions))})")
        return "\n".join(lines)

    def _md_enms_performance(self, data: Dict[str, Any]) -> str:
        """Render EnMS performance summary (Clause 9.3c)."""
        perf = data.get("enms_performance", {})
        return (
            "## 3. EnMS Performance Summary (9.3c)\n\n"
            f"**Total Energy Consumption:** {self._fmt(perf.get('total_consumption_mwh', 0))} MWh  \n"
            f"**Baseline Consumption:** {self._fmt(perf.get('baseline_consumption_mwh', 0))} MWh  \n"
            f"**Absolute Change:** {self._fmt(perf.get('absolute_change_mwh', 0))} MWh  \n"
            f"**Normalized Change:** {self._fmt(perf.get('normalized_change_pct', 0))}%  \n"
            f"**Energy Cost:** EUR {self._fmt(perf.get('total_cost_eur', 0))}  \n"
            f"**SEU Coverage:** {self._fmt(perf.get('seu_coverage_pct', 0))}%  \n"
            f"**Data Collection Compliance:** "
            f"{self._fmt(perf.get('data_compliance_pct', 0))}%"
        )

    def _md_enpi_trends(self, data: Dict[str, Any]) -> str:
        """Render EnPI trends and analysis (Clause 9.3c)."""
        enpi = data.get("enpi_trends", {})
        indicators = enpi.get("indicators", [])
        lines = ["## 4. EnPI Trends & Analysis (9.3c)\n"]
        if indicators:
            lines.extend([
                "| EnPI | Unit | Baseline | Current | Target | Variance | Status |",
                "|------|------|----------|---------|--------|----------|--------|",
            ])
            for ind in indicators:
                lines.append(
                    f"| {ind.get('name', '-')} "
                    f"| {ind.get('unit', '-')} "
                    f"| {self._fmt(ind.get('baseline_value', 0))} "
                    f"| {self._fmt(ind.get('current_value', 0))} "
                    f"| {self._fmt(ind.get('target_value', 0))} "
                    f"| {self._fmt(ind.get('variance_pct', 0))}% "
                    f"| {ind.get('status', '-')} |"
                )
        trend_data = enpi.get("monthly_trend", [])
        if trend_data:
            lines.extend([
                "\n### Monthly EnPI Trend\n",
                "| Month | Primary EnPI | Target | Cumulative Savings (MWh) |",
                "|-------|-------------|--------|-------------------------|",
            ])
            for t in trend_data[-12:]:
                lines.append(
                    f"| {t.get('month', '-')} "
                    f"| {self._fmt(t.get('enpi_value', 0))} "
                    f"| {self._fmt(t.get('target', 0))} "
                    f"| {self._fmt(t.get('cumulative_savings_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_objectives_targets(self, data: Dict[str, Any]) -> str:
        """Render objectives and targets status (Clause 9.3e)."""
        objectives = data.get("objectives_targets", [])
        if not objectives:
            return "## 5. Objectives & Targets Status (9.3e)\n\n_No objectives defined._"
        lines = [
            "## 5. Objectives & Targets Status (9.3e)\n",
            "| Objective | Target | Current | Progress | Deadline | Status |",
            "|-----------|--------|---------|----------|----------|--------|",
        ]
        for obj in objectives:
            progress_bar = self._progress_bar(obj.get("progress_pct", 0))
            lines.append(
                f"| {obj.get('objective', '-')} "
                f"| {obj.get('target_value', '-')} "
                f"| {obj.get('current_value', '-')} "
                f"| {progress_bar} {self._fmt(obj.get('progress_pct', 0))}% "
                f"| {obj.get('deadline', '-')} "
                f"| {obj.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_internal_audit(self, data: Dict[str, Any]) -> str:
        """Render internal audit results (Clause 9.3f)."""
        audit = data.get("internal_audit", {})
        findings = audit.get("findings", [])
        lines = [
            "## 6. Internal Audit Results (9.3f)\n",
            f"**Last Audit Date:** {audit.get('last_audit_date', '-')}  ",
            f"**Auditor:** {audit.get('auditor', '-')}  ",
            f"**Scope:** {audit.get('scope', '-')}  ",
            f"**Total Findings:** {audit.get('total_findings', 0)}  ",
            f"**Major:** {audit.get('major_findings', 0)}  ",
            f"**Minor:** {audit.get('minor_findings', 0)}  ",
            f"**Observations:** {audit.get('observations', 0)}  ",
            f"**Closed:** {audit.get('closed_findings', 0)}",
        ]
        if findings:
            lines.extend([
                "\n### Findings Summary\n",
                "| # | Clause | Finding | Severity | Status | Due Date |",
                "|---|--------|---------|----------|--------|----------|",
            ])
            for i, f in enumerate(findings, 1):
                lines.append(
                    f"| {i} | {f.get('clause', '-')} "
                    f"| {f.get('finding', '-')} "
                    f"| {f.get('severity', '-')} "
                    f"| {f.get('status', '-')} "
                    f"| {f.get('due_date', '-')} |"
                )
        return "\n".join(lines)

    def _md_nonconformities(self, data: Dict[str, Any]) -> str:
        """Render nonconformities and corrective actions (Clause 9.3g)."""
        ncs = data.get("nonconformities", [])
        if not ncs:
            return (
                "## 7. Nonconformities & Corrective Actions (9.3g)\n\n"
                "_No open nonconformities._"
            )
        lines = [
            "## 7. Nonconformities & Corrective Actions (9.3g)\n",
            "| NC # | Description | Root Cause | Corrective Action | "
            "Owner | Due Date | Status |",
            "|------|-----------|-----------|-------------------|"
            "-------|----------|--------|",
        ]
        for nc in ncs:
            lines.append(
                f"| {nc.get('nc_id', '-')} "
                f"| {nc.get('description', '-')} "
                f"| {nc.get('root_cause', '-')} "
                f"| {nc.get('corrective_action', '-')} "
                f"| {nc.get('owner', '-')} "
                f"| {nc.get('due_date', '-')} "
                f"| {nc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_energy_policy(self, data: Dict[str, Any]) -> str:
        """Render energy policy review section."""
        policy = data.get("energy_policy", {})
        lines = [
            "## 8. Energy Policy Review\n",
            f"**Policy Last Reviewed:** {policy.get('last_reviewed', '-')}  ",
            f"**Policy Version:** {policy.get('version', '-')}  ",
            f"**Revision Needed:** {policy.get('revision_needed', 'No')}  ",
            f"**Alignment with Strategy:** {policy.get('strategy_alignment', '-')}  ",
            f"**Communication Status:** {policy.get('communication_status', '-')}",
        ]
        changes = policy.get("proposed_changes", [])
        if changes:
            lines.append("\n### Proposed Changes\n")
            for c in changes:
                lines.append(f"- {c}")
        return "\n".join(lines)

    def _md_continual_improvement(self, data: Dict[str, Any]) -> str:
        """Render continual improvement evidence section."""
        ci = data.get("continual_improvement", {})
        projects = ci.get("projects", [])
        lines = [
            "## 9. Continual Improvement Evidence\n",
            f"**Total Projects Completed:** {ci.get('completed_projects', 0)}  ",
            f"**Total Energy Saved:** {self._fmt(ci.get('total_savings_mwh', 0))} MWh  ",
            f"**Total Cost Savings:** EUR {self._fmt(ci.get('total_cost_savings_eur', 0))}  ",
            f"**Total Investment:** EUR {self._fmt(ci.get('total_investment_eur', 0))}  ",
            f"**Year-over-Year Improvement:** {self._fmt(ci.get('yoy_improvement_pct', 0))}%",
        ]
        if projects:
            lines.extend([
                "\n### Improvement Projects\n",
                "| Project | Category | Savings (MWh/yr) | Cost (EUR) | Status |",
                "|---------|----------|-----------------|-----------|--------|",
            ])
            for p in projects:
                lines.append(
                    f"| {p.get('name', '-')} "
                    f"| {p.get('category', '-')} "
                    f"| {self._fmt(p.get('savings_mwh', 0))} "
                    f"| {self._fmt(p.get('cost_eur', 0))} "
                    f"| {p.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_management_decisions(self, data: Dict[str, Any]) -> str:
        """Render management decisions required section."""
        decisions = data.get("management_decisions", [])
        if not decisions:
            return "## 10. Management Decisions Required\n\n_No decisions pending._"
        lines = [
            "## 10. Management Decisions Required\n",
            "| # | Decision | Context | Recommended Action | Priority |",
            "|---|---------|---------|-------------------|----------|",
        ]
        for i, d in enumerate(decisions, 1):
            lines.append(
                f"| {i} | {d.get('decision', '-')} "
                f"| {d.get('context', '-')} "
                f"| {d.get('recommendation', '-')} "
                f"| {d.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>ISO 50001 Management Review Package</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'Standard: ISO 50001:2018 Clause 9.3</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">EnMS Status</span>'
            f'<span class="value">{s.get("enms_status", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Improvement</span>'
            f'<span class="value">{self._fmt(s.get("improvement_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Objectives Met</span>'
            f'<span class="value">{s.get("objectives_met", 0)}/{s.get("total_objectives", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Open NCs</span>'
            f'<span class="value">{s.get("open_nonconformities", 0)}</span></div>\n'
            '</div>'
        )

    def _html_enpi_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI trends."""
        enpi = data.get("enpi_trends", {})
        indicators = enpi.get("indicators", [])
        rows = ""
        for ind in indicators:
            status_color = "#059669" if ind.get("status") == "on_track" else "#d97706"
            rows += (
                f'<tr><td>{ind.get("name", "-")}</td>'
                f'<td>{self._fmt(ind.get("current_value", 0))} {ind.get("unit", "")}</td>'
                f'<td>{self._fmt(ind.get("target_value", 0))}</td>'
                f'<td style="color:{status_color};">{ind.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>EnPI Trends</h2>\n<table>\n'
            '<tr><th>EnPI</th><th>Current</th><th>Target</th><th>Status</th></tr>\n'
            f'{rows}</table>\n'
            '<div class="chart-placeholder" data-chart="enpi_trend">[EnPI Chart]</div>'
        )

    def _html_objectives_targets(self, data: Dict[str, Any]) -> str:
        """Render HTML objectives and targets."""
        objectives = data.get("objectives_targets", [])
        bars = ""
        for obj in objectives:
            pct = min(obj.get("progress_pct", 0), 100)
            color = "#059669" if obj.get("status") == "achieved" else "#0d6efd"
            bars += (
                f'<div class="target-row">'
                f'<span class="target-name">{obj.get("objective", "-")}</span>'
                f'<div class="progress-bar"><div class="progress-fill" '
                f'style="width:{pct}%;background:{color};"></div></div>'
                f'<span class="target-pct">{self._fmt(pct)}%</span></div>\n'
            )
        return f'<h2>Objectives & Targets</h2>\n{bars}'

    def _html_audit_results(self, data: Dict[str, Any]) -> str:
        """Render HTML audit results."""
        audit = data.get("internal_audit", {})
        return (
            '<h2>Internal Audit Results</h2>\n'
            f'<p>Findings: {audit.get("total_findings", 0)} '
            f'(Major: {audit.get("major_findings", 0)}, '
            f'Minor: {audit.get("minor_findings", 0)}) | '
            f'Closed: {audit.get("closed_findings", 0)}</p>'
        )

    def _html_nonconformities(self, data: Dict[str, Any]) -> str:
        """Render HTML nonconformities."""
        ncs = data.get("nonconformities", [])
        if not ncs:
            return '<h2>Nonconformities</h2>\n<p class="success">No open nonconformities</p>'
        rows = ""
        for nc in ncs:
            rows += (
                f'<tr><td>{nc.get("nc_id", "-")}</td>'
                f'<td>{nc.get("description", "-")}</td>'
                f'<td>{nc.get("owner", "-")}</td>'
                f'<td>{nc.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Nonconformities</h2>\n<table>\n'
            '<tr><th>NC#</th><th>Description</th><th>Owner</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_improvement_evidence(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement evidence."""
        ci = data.get("continual_improvement", {})
        return (
            '<h2>Continual Improvement</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Projects</span>'
            f'<span class="value">{ci.get("completed_projects", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Savings</span>'
            f'<span class="value">{self._fmt(ci.get("total_savings_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">EUR {self._fmt(ci.get("total_cost_savings_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">YoY Improvement</span>'
            f'<span class="value">{self._fmt(ci.get("yoy_improvement_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_management_decisions(self, data: Dict[str, Any]) -> str:
        """Render HTML management decisions."""
        decisions = data.get("management_decisions", [])
        items = "".join(
            f'<li><strong>[{d.get("priority", "-")}]</strong> {d.get("decision", "-")} - '
            f'{d.get("recommendation", "-")}</li>\n'
            for d in decisions
        )
        return f'<h2>Management Decisions Required</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return data.get("executive_summary", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        enpi = data.get("enpi_trends", {}).get("monthly_trend", [])
        objectives = data.get("objectives_targets", [])
        ci = data.get("continual_improvement", {}).get("projects", [])
        return {
            "enpi_monthly_trend": {
                "type": "line",
                "labels": [t.get("month", "") for t in enpi],
                "series": {
                    "actual": [t.get("enpi_value", 0) for t in enpi],
                    "target": [t.get("target", 0) for t in enpi],
                },
            },
            "objective_progress": {
                "type": "horizontal_bar",
                "labels": [o.get("objective", "") for o in objectives],
                "values": [o.get("progress_pct", 0) for o in objectives],
                "target": 100,
            },
            "improvement_projects": {
                "type": "bar",
                "labels": [p.get("name", "") for p in ci],
                "values": [p.get("savings_mwh", 0) for p in ci],
            },
            "cumulative_savings": {
                "type": "area",
                "labels": [t.get("month", "") for t in enpi],
                "values": [t.get("cumulative_savings_mwh", 0) for t in enpi],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _progress_bar(self, pct: float, width: int = 15) -> str:
        """Generate a text-based progress bar.

        Args:
            pct: Progress percentage (0-100).
            width: Character width of the bar.

        Returns:
            Text progress bar string.
        """
        filled = min(int(pct / 100 * width), width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".target-row{display:flex;align-items:center;gap:10px;margin:8px 0;}"
            ".target-name{width:250px;font-weight:600;}"
            ".progress-bar{flex:1;height:20px;background:#e9ecef;border-radius:4px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:4px;}"
            ".target-pct{width:60px;text-align:right;font-weight:600;}"
            ".chart-placeholder{background:#f0f0f0;border:2px dashed #ccc;padding:40px;"
            "text-align:center;margin:15px 0;}"
            ".success{color:#059669;font-style:italic;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
