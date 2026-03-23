# -*- coding: utf-8 -*-
"""
ISO50001ComplianceReportTemplate - ISO 50001 management review for PACK-039.

Generates comprehensive ISO 50001 compliance reports for management
review meetings showing EnPI results against objectives, energy
objectives and targets progress, energy baseline status, improvement
action register, and conformity assessment summary.

Sections:
    1. Management Review Summary
    2. EnPI Results
    3. Objectives and Targets Progress
    4. Energy Baseline Status
    5. Significant Energy Uses (SEUs)
    6. Improvement Action Register
    7. Conformity Assessment
    8. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management systems - Requirements)
    - ISO 50006:2014 (Measuring energy performance using EnPIs and EnBs)
    - ISO 50015:2014 (M&V of energy performance of organizations)

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


class ISO50001ComplianceReportTemplate:
    """
    ISO 50001 compliance report template.

    Renders ISO 50001 management review reports showing EnPI results,
    objectives and targets progress, energy baseline status, significant
    energy use analysis, improvement action register, and conformity
    assessment across markdown, HTML, and JSON formats. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ISO50001ComplianceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ISO 50001 compliance report as Markdown.

        Args:
            data: ISO 50001 compliance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_management_review(data),
            self._md_enpi_results(data),
            self._md_objectives_progress(data),
            self._md_baseline_status(data),
            self._md_seus(data),
            self._md_improvement_actions(data),
            self._md_conformity_assessment(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ISO 50001 compliance report as self-contained HTML.

        Args:
            data: ISO 50001 compliance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_management_review(data),
            self._html_enpi_results(data),
            self._html_objectives_progress(data),
            self._html_baseline_status(data),
            self._html_seus(data),
            self._html_improvement_actions(data),
            self._html_conformity_assessment(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ISO 50001 Compliance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ISO 50001 compliance report as structured JSON.

        Args:
            data: ISO 50001 compliance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "iso50001_compliance_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "management_review": self._json_management_review(data),
            "enpi_results": data.get("enpi_results", []),
            "objectives_progress": data.get("objectives_progress", []),
            "baseline_status": data.get("baseline_status", {}),
            "seus": data.get("seus", []),
            "improvement_actions": data.get("improvement_actions", []),
            "conformity_assessment": data.get("conformity_assessment", {}),
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
            f"# ISO 50001 Compliance Report\n\n"
            f"**Organization:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Certification Status:** {data.get('certification_status', '')}  \n"
            f"**Next Audit Date:** {data.get('next_audit_date', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 ISO50001ComplianceReportTemplate v39.0.0\n\n---"
        )

    def _md_management_review(self, data: Dict[str, Any]) -> str:
        """Render management review summary section."""
        review = data.get("management_review", {})
        return (
            "## 1. Management Review Summary\n\n"
            "| Item | Status |\n|------|--------|\n"
            f"| Review Date | {review.get('review_date', '-')} |\n"
            f"| Attendees | {review.get('attendees', '-')} |\n"
            f"| EnMS Scope | {review.get('enms_scope', '-')} |\n"
            f"| Overall Energy Performance | {review.get('overall_performance', '-')} |\n"
            f"| Energy Policy Status | {review.get('policy_status', '-')} |\n"
            f"| Nonconformities Open | {self._fmt(review.get('nonconformities_open', 0), 0)} |\n"
            f"| Nonconformities Closed | {self._fmt(review.get('nonconformities_closed', 0), 0)} |\n"
            f"| Continual Improvement Rating | {review.get('improvement_rating', '-')} |"
        )

    def _md_enpi_results(self, data: Dict[str, Any]) -> str:
        """Render EnPI results section."""
        enpis = data.get("enpi_results", [])
        if not enpis:
            return "## 2. EnPI Results\n\n_No EnPI results available._"
        lines = [
            "## 2. EnPI Results\n",
            "| EnPI | Unit | Baseline | Current | Target | Improvement | Status |",
            "|------|------|--------:|---------:|-------:|----------:|--------|",
        ]
        for e in enpis:
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {e.get('unit', '-')} "
                f"| {self._fmt(e.get('baseline', 0), 3)} "
                f"| {self._fmt(e.get('current', 0), 3)} "
                f"| {self._fmt(e.get('target', 0), 3)} "
                f"| {self._fmt(e.get('improvement_pct', 0))}% "
                f"| {e.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_objectives_progress(self, data: Dict[str, Any]) -> str:
        """Render objectives and targets progress section."""
        objectives = data.get("objectives_progress", [])
        if not objectives:
            return "## 3. Objectives and Targets Progress\n\n_No objectives data available._"
        lines = [
            "## 3. Objectives and Targets Progress\n",
            "| Objective | Target | Progress | Deadline | Owner | Status |",
            "|-----------|--------|--------:|----------|-------|--------|",
        ]
        for obj in objectives:
            lines.append(
                f"| {obj.get('objective', '-')} "
                f"| {obj.get('target', '-')} "
                f"| {self._fmt(obj.get('progress_pct', 0))}% "
                f"| {obj.get('deadline', '-')} "
                f"| {obj.get('owner', '-')} "
                f"| {obj.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_baseline_status(self, data: Dict[str, Any]) -> str:
        """Render energy baseline status section."""
        baseline = data.get("baseline_status", {})
        if not baseline:
            return "## 4. Energy Baseline Status\n\n_No baseline status data available._"
        adjustments = baseline.get("adjustments", [])
        lines = [
            "## 4. Energy Baseline Status\n",
            f"**Baseline Period:** {baseline.get('baseline_period', '-')}  \n"
            f"**Baseline Consumption:** {self._format_energy(baseline.get('baseline_consumption_mwh', 0))}  \n"
            f"**Last Reviewed:** {baseline.get('last_reviewed', '-')}  \n"
            f"**Baseline Valid:** {baseline.get('is_valid', '-')}  \n"
            f"**Revision Required:** {baseline.get('revision_required', '-')}\n",
        ]
        if adjustments:
            lines.append("### Baseline Adjustments\n")
            lines.append("| Factor | Type | Impact (MWh) | Applied |")
            lines.append("|--------|------|------------:|---------|")
            for adj in adjustments:
                lines.append(
                    f"| {adj.get('factor', '-')} "
                    f"| {adj.get('type', '-')} "
                    f"| {self._fmt(adj.get('impact_mwh', 0), 1)} "
                    f"| {adj.get('applied', '-')} |"
                )
        return "\n".join(lines)

    def _md_seus(self, data: Dict[str, Any]) -> str:
        """Render significant energy uses section."""
        seus = data.get("seus", [])
        if not seus:
            return "## 5. Significant Energy Uses (SEUs)\n\n_No SEU data available._"
        lines = [
            "## 5. Significant Energy Uses (SEUs)\n",
            "| SEU | Consumption (MWh) | % of Total | Trend | Performance | Action Plan |",
            "|-----|------------------:|----------:|-------|------------|-------------|",
        ]
        total = data.get("total_consumption_mwh", 1)
        for seu in seus:
            cons = seu.get("consumption_mwh", 0)
            lines.append(
                f"| {seu.get('name', '-')} "
                f"| {self._fmt(cons, 1)} "
                f"| {self._pct(cons, total)} "
                f"| {seu.get('trend', '-')} "
                f"| {seu.get('performance', '-')} "
                f"| {seu.get('action_plan', '-')} |"
            )
        return "\n".join(lines)

    def _md_improvement_actions(self, data: Dict[str, Any]) -> str:
        """Render improvement action register section."""
        actions = data.get("improvement_actions", [])
        if not actions:
            return "## 6. Improvement Action Register\n\n_No improvement actions registered._"
        lines = [
            "## 6. Improvement Action Register\n",
            "| # | Action | Type | Owner | Due Date | Savings (MWh/yr) | Status |",
            "|---|--------|------|-------|----------|----------------:|--------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('type', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {self._fmt(a.get('savings_mwh_yr', 0), 1)} "
                f"| {a.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_conformity_assessment(self, data: Dict[str, Any]) -> str:
        """Render conformity assessment section."""
        conform = data.get("conformity_assessment", {})
        if not conform:
            return "## 7. Conformity Assessment\n\n_No conformity data available._"
        clauses = conform.get("clauses", [])
        lines = [
            "## 7. Conformity Assessment\n",
            f"**Overall Conformity:** {conform.get('overall_conformity', '-')}  \n"
            f"**Major Nonconformities:** {self._fmt(conform.get('major_nc', 0), 0)}  \n"
            f"**Minor Nonconformities:** {self._fmt(conform.get('minor_nc', 0), 0)}  \n"
            f"**Observations:** {self._fmt(conform.get('observations', 0), 0)}\n",
        ]
        if clauses:
            lines.append("### Clause Assessment\n")
            lines.append("| Clause | Title | Conformity | Findings |")
            lines.append("|--------|-------|------------|----------|")
            for cl in clauses:
                lines.append(
                    f"| {cl.get('clause', '-')} "
                    f"| {cl.get('title', '-')} "
                    f"| {cl.get('conformity', '-')} "
                    f"| {cl.get('findings', '-')} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Close open nonconformities before next surveillance audit",
                "Update energy baseline to reflect recent static factor changes",
                "Strengthen SEU monitoring with additional sub-metering",
                "Review and update energy objectives for next planning cycle",
            ]
        lines = ["## 8. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

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
            f'<h1>ISO 50001 Compliance Report</h1>\n'
            f'<p class="subtitle">Organization: {facility} | '
            f'Status: {data.get("certification_status", "-")} | '
            f'Period: {data.get("reporting_period", "-")}</p>'
        )

    def _html_management_review(self, data: Dict[str, Any]) -> str:
        """Render HTML management review cards."""
        r = data.get("management_review", {})
        nc_open = r.get("nonconformities_open", 0)
        nc_cls = "severity-high" if nc_open > 0 else "severity-low"
        return (
            '<h2>Management Review Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Performance</span>'
            f'<span class="value">{r.get("overall_performance", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Policy Status</span>'
            f'<span class="value">{r.get("policy_status", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">NC Open</span>'
            f'<span class="value {nc_cls}">{nc_open}</span></div>\n'
            f'  <div class="card"><span class="label">NC Closed</span>'
            f'<span class="value">{r.get("nonconformities_closed", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Improvement</span>'
            f'<span class="value">{r.get("improvement_rating", "-")}</span></div>\n'
            '</div>'
        )

    def _html_enpi_results(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI results table."""
        enpis = data.get("enpi_results", [])
        rows = ""
        for e in enpis:
            status = e.get("status", "").lower()
            cls = "severity-high" if status in ("below target", "fail") else (
                "severity-low" if status in ("on target", "pass") else "")
            rows += (
                f'<tr><td>{e.get("name", "-")}</td>'
                f'<td>{self._fmt(e.get("baseline", 0), 3)}</td>'
                f'<td>{self._fmt(e.get("current", 0), 3)}</td>'
                f'<td>{self._fmt(e.get("target", 0), 3)}</td>'
                f'<td>{self._fmt(e.get("improvement_pct", 0))}%</td>'
                f'<td class="{cls}">{e.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>EnPI Results</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Baseline</th><th>Current</th>'
            f'<th>Target</th><th>Improvement</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_objectives_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML objectives progress table."""
        objectives = data.get("objectives_progress", [])
        rows = ""
        for obj in objectives:
            progress = obj.get("progress_pct", 0)
            cls = "severity-high" if progress < 50 else (
                "severity-medium" if progress < 80 else "severity-low")
            rows += (
                f'<tr><td>{obj.get("objective", "-")}</td>'
                f'<td>{obj.get("target", "-")}</td>'
                f'<td class="{cls}">{self._fmt(progress)}%</td>'
                f'<td>{obj.get("deadline", "-")}</td>'
                f'<td>{obj.get("owner", "-")}</td>'
                f'<td>{obj.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Objectives and Targets</h2>\n'
            '<table>\n<tr><th>Objective</th><th>Target</th><th>Progress</th>'
            f'<th>Deadline</th><th>Owner</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_baseline_status(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline status section."""
        baseline = data.get("baseline_status", {})
        valid = baseline.get("is_valid", "")
        cls = "severity-low" if valid == "Yes" else "severity-high"
        return (
            '<h2>Energy Baseline Status</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Baseline Period</span>'
            f'<span class="value">{baseline.get("baseline_period", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Consumption</span>'
            f'<span class="value">{self._fmt(baseline.get("baseline_consumption_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Valid</span>'
            f'<span class="value {cls}">{valid}</span></div>\n'
            f'  <div class="card"><span class="label">Last Reviewed</span>'
            f'<span class="value">{baseline.get("last_reviewed", "-")}</span></div>\n'
            '</div>'
        )

    def _html_seus(self, data: Dict[str, Any]) -> str:
        """Render HTML SEU table."""
        seus = data.get("seus", [])
        total = data.get("total_consumption_mwh", 1)
        rows = ""
        for seu in seus:
            cons = seu.get("consumption_mwh", 0)
            rows += (
                f'<tr><td>{seu.get("name", "-")}</td>'
                f'<td>{self._fmt(cons, 1)}</td>'
                f'<td>{self._pct(cons, total)}</td>'
                f'<td>{seu.get("trend", "-")}</td>'
                f'<td>{seu.get("performance", "-")}</td></tr>\n'
            )
        return (
            '<h2>Significant Energy Uses (SEUs)</h2>\n'
            '<table>\n<tr><th>SEU</th><th>Consumption (MWh)</th><th>% of Total</th>'
            f'<th>Trend</th><th>Performance</th></tr>\n{rows}</table>'
        )

    def _html_improvement_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement actions table."""
        actions = data.get("improvement_actions", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            status = a.get("status", "").lower()
            cls = "severity-high" if status == "overdue" else ""
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("type", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td>{a.get("due_date", "-")}</td>'
                f'<td>{self._fmt(a.get("savings_mwh_yr", 0), 1)}</td>'
                f'<td class="{cls}">{a.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Improvement Action Register</h2>\n'
            '<table>\n<tr><th>#</th><th>Action</th><th>Type</th><th>Owner</th>'
            f'<th>Due Date</th><th>Savings (MWh/yr)</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_conformity_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML conformity assessment section."""
        conform = data.get("conformity_assessment", {})
        clauses = conform.get("clauses", [])
        rows = ""
        for cl in clauses:
            conf = cl.get("conformity", "").lower()
            cls = "severity-high" if conf == "nonconforming" else (
                "severity-medium" if conf == "observation" else "")
            rows += (
                f'<tr><td>{cl.get("clause", "-")}</td>'
                f'<td>{cl.get("title", "-")}</td>'
                f'<td class="{cls}">{cl.get("conformity", "-")}</td>'
                f'<td>{cl.get("findings", "-")}</td></tr>\n'
            )
        return (
            '<h2>Conformity Assessment</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Overall</span>'
            f'<span class="value">{conform.get("overall_conformity", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Major NCs</span>'
            f'<span class="value severity-high">{conform.get("major_nc", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Minor NCs</span>'
            f'<span class="value severity-medium">{conform.get("minor_nc", 0)}</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Clause</th><th>Title</th>'
            f'<th>Conformity</th><th>Findings</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Close open nonconformities before next surveillance audit",
            "Update energy baseline to reflect recent static factor changes",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_management_review(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON management review summary."""
        r = data.get("management_review", {})
        return {
            "review_date": r.get("review_date", ""),
            "overall_performance": r.get("overall_performance", ""),
            "policy_status": r.get("policy_status", ""),
            "nonconformities_open": r.get("nonconformities_open", 0),
            "nonconformities_closed": r.get("nonconformities_closed", 0),
            "improvement_rating": r.get("improvement_rating", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        enpis = data.get("enpi_results", [])
        objectives = data.get("objectives_progress", [])
        seus = data.get("seus", [])
        return {
            "enpi_comparison": {
                "type": "grouped_bar",
                "labels": [e.get("name", "") for e in enpis],
                "series": {
                    "baseline": [e.get("baseline", 0) for e in enpis],
                    "current": [e.get("current", 0) for e in enpis],
                    "target": [e.get("target", 0) for e in enpis],
                },
            },
            "objectives_progress": {
                "type": "bar",
                "labels": [o.get("objective", "") for o in objectives],
                "values": [o.get("progress_pct", 0) for o in objectives],
            },
            "seu_breakdown": {
                "type": "pie",
                "labels": [s.get("name", "") for s in seus],
                "values": [s.get("consumption_mwh", 0) for s in seus],
            },
            "conformity_gauge": {
                "type": "gauge",
                "value": data.get("conformity_assessment", {}).get("conformity_score", 0),
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
