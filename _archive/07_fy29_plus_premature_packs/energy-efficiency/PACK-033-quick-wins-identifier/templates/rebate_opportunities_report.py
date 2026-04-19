# -*- coding: utf-8 -*-
"""
RebateOpportunitiesReportTemplate - Utility rebate report for PACK-033.

Generates utility and government rebate opportunity reports for quick-win
energy efficiency measures, including program matching, application status,
timeline management, net cost impact analysis, and rebate stacking analysis.

Sections:
    1. Rebate Summary
    2. Program Matches
    3. Application Status
    4. Timeline
    5. Net Cost Impact
    6. Stacking Analysis

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RebateOpportunitiesReportTemplate:
    """
    Utility rebate opportunities report template.

    Renders rebate program matching results, application tracking,
    net cost impact calculations, and stacking analysis across
    markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RebateOpportunitiesReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render rebate opportunities report as Markdown.

        Args:
            data: Rebate analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_rebate_summary(data),
            self._md_program_matches(data),
            self._md_application_status(data),
            self._md_timeline(data),
            self._md_net_cost_impact(data),
            self._md_stacking_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render rebate opportunities report as self-contained HTML.

        Args:
            data: Rebate analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_rebate_summary(data),
            self._html_program_matches(data),
            self._html_application_status(data),
            self._html_net_cost_impact(data),
            self._html_stacking_analysis(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Rebate Opportunities Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render rebate opportunities report as structured JSON.

        Args:
            data: Rebate analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "rebate_opportunities_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "rebate_summary": self._json_rebate_summary(data),
            "program_matches": data.get("program_matches", []),
            "application_status": data.get("application_status", []),
            "timeline": data.get("timeline", []),
            "net_cost_impact": data.get("net_cost_impact", {}),
            "stacking_analysis": data.get("stacking_analysis", []),
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
            f"# Rebate Opportunities Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Utility Territory:** {data.get('utility_territory', '')}  \n"
            f"**Analysis Date:** {data.get('analysis_date', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 RebateOpportunitiesReportTemplate v33.0.0\n\n---"
        )

    def _md_rebate_summary(self, data: Dict[str, Any]) -> str:
        """Render rebate summary section."""
        summary = data.get("rebate_summary", {})
        return (
            "## 1. Rebate Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Programs Matched | {summary.get('programs_matched', 0)} |\n"
            f"| Total Potential Rebates | {self._format_currency(summary.get('total_potential_rebates', 0))} |\n"
            f"| Rebates Applied For | {self._format_currency(summary.get('rebates_applied', 0))} |\n"
            f"| Rebates Received | {self._format_currency(summary.get('rebates_received', 0))} |\n"
            f"| Rebates Pending | {self._format_currency(summary.get('rebates_pending', 0))} |\n"
            f"| Coverage Rate | {self._fmt(summary.get('coverage_rate_pct', 0))}% |\n"
            f"| Net Investment Reduction | {self._fmt(summary.get('net_investment_reduction_pct', 0))}% |"
        )

    def _md_program_matches(self, data: Dict[str, Any]) -> str:
        """Render program matches table."""
        matches = data.get("program_matches", [])
        if not matches:
            return "## 2. Program Matches\n\n_No matching rebate programs found._"
        lines = [
            "## 2. Program Matches\n",
            "| Utility | Program | Measure | Rebate Amount | Type | Deadline |",
            "|---------|---------|---------|---------------|------|----------|",
        ]
        for m in matches:
            lines.append(
                f"| {m.get('utility', '-')} "
                f"| {m.get('program', '-')} "
                f"| {m.get('measure', '-')} "
                f"| {self._format_currency(m.get('rebate_amount', 0))} "
                f"| {m.get('rebate_type', '-')} "
                f"| {m.get('deadline', '-')} |"
            )
        total = sum(m.get("rebate_amount", 0) for m in matches)
        lines.append(
            f"| | | **TOTAL** | **{self._format_currency(total)}** | | |"
        )
        return "\n".join(lines)

    def _md_application_status(self, data: Dict[str, Any]) -> str:
        """Render application status section."""
        statuses = data.get("application_status", [])
        if not statuses:
            return "## 3. Application Status\n\n_No applications submitted._"
        lines = [
            "## 3. Application Status\n",
            "| Program | Measure | Status | Submitted | Expected Decision | Amount |",
            "|---------|---------|--------|-----------|-------------------|--------|",
        ]
        for s in statuses:
            lines.append(
                f"| {s.get('program', '-')} "
                f"| {s.get('measure', '-')} "
                f"| {s.get('status', '-')} "
                f"| {s.get('submitted_date', '-')} "
                f"| {s.get('expected_decision', '-')} "
                f"| {self._format_currency(s.get('amount', 0))} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render rebate timeline section."""
        timeline = data.get("timeline", [])
        if not timeline:
            return "## 4. Timeline\n\n_No timeline data available._"
        lines = [
            "## 4. Timeline\n",
            "| Date | Event | Program | Action Required |",
            "|------|-------|---------|----------------|",
        ]
        for t in timeline:
            lines.append(
                f"| {t.get('date', '-')} "
                f"| {t.get('event', '-')} "
                f"| {t.get('program', '-')} "
                f"| {t.get('action_required', '-')} |"
            )
        return "\n".join(lines)

    def _md_net_cost_impact(self, data: Dict[str, Any]) -> str:
        """Render net cost impact section."""
        impact = data.get("net_cost_impact", {})
        measures = impact.get("by_measure", [])
        if not measures:
            lines = [
                "## 5. Net Cost Impact\n",
                f"- **Gross Investment:** {self._format_currency(impact.get('gross_investment', 0))}",
                f"- **Total Rebates:** {self._format_currency(impact.get('total_rebates', 0))}",
                f"- **Net Investment:** {self._format_currency(impact.get('net_investment', 0))}",
                f"- **Effective Payback (with rebates):** "
                f"{self._fmt(impact.get('effective_payback_months', 0), 1)} months",
            ]
            return "\n".join(lines)
        lines = [
            "## 5. Net Cost Impact\n",
            f"**Gross Investment:** {self._format_currency(impact.get('gross_investment', 0))}  ",
            f"**Total Rebates:** {self._format_currency(impact.get('total_rebates', 0))}  ",
            f"**Net Investment:** {self._format_currency(impact.get('net_investment', 0))}  ",
            f"**Effective Payback:** {self._fmt(impact.get('effective_payback_months', 0), 1)} months\n",
            "| Measure | Gross Cost | Rebates | Net Cost | Original Payback | New Payback |",
            "|---------|-----------|---------|----------|-----------------|------------|",
        ]
        for m in measures:
            lines.append(
                f"| {m.get('measure', '-')} "
                f"| {self._format_currency(m.get('gross_cost', 0))} "
                f"| {self._format_currency(m.get('rebates', 0))} "
                f"| {self._format_currency(m.get('net_cost', 0))} "
                f"| {self._fmt(m.get('original_payback_months', 0), 1)} mo "
                f"| {self._fmt(m.get('new_payback_months', 0), 1)} mo |"
            )
        return "\n".join(lines)

    def _md_stacking_analysis(self, data: Dict[str, Any]) -> str:
        """Render rebate stacking analysis section."""
        stacking = data.get("stacking_analysis", [])
        if not stacking:
            return "## 6. Stacking Analysis\n\n_No stacking opportunities identified._"
        lines = [
            "## 6. Stacking Analysis\n",
            "Rebate stacking combines multiple incentive programs for the same measure.\n",
            "| Measure | Programs Stacked | Combined Rebate | Cost Coverage (%) | Stackable |",
            "|---------|-----------------|----------------|-------------------|-----------|",
        ]
        for s in stacking:
            programs = ", ".join(s.get("programs", []))
            lines.append(
                f"| {s.get('measure', '-')} "
                f"| {programs} "
                f"| {self._format_currency(s.get('combined_rebate', 0))} "
                f"| {self._fmt(s.get('cost_coverage_pct', 0))}% "
                f"| {'Yes' if s.get('stackable', False) else 'No'} |"
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
            f'<h1>Rebate Opportunities Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Territory: {data.get("utility_territory", "-")}</p>'
        )

    def _html_rebate_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML rebate summary cards."""
        s = data.get("rebate_summary", {})
        return (
            '<h2>Rebate Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Programs Matched</span>'
            f'<span class="value">{s.get("programs_matched", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Potential Rebates</span>'
            f'<span class="value">{self._format_currency(s.get("total_potential_rebates", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Received</span>'
            f'<span class="value">{self._format_currency(s.get("rebates_received", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Coverage</span>'
            f'<span class="value">{self._fmt(s.get("coverage_rate_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_program_matches(self, data: Dict[str, Any]) -> str:
        """Render HTML program matches table."""
        matches = data.get("program_matches", [])
        rows = ""
        for m in matches:
            rows += (
                f'<tr><td>{m.get("utility", "-")}</td>'
                f'<td>{m.get("program", "-")}</td>'
                f'<td>{m.get("measure", "-")}</td>'
                f'<td>{self._format_currency(m.get("rebate_amount", 0))}</td>'
                f'<td>{m.get("deadline", "-")}</td></tr>\n'
            )
        return (
            '<h2>Program Matches</h2>\n'
            '<table>\n<tr><th>Utility</th><th>Program</th><th>Measure</th>'
            f'<th>Rebate</th><th>Deadline</th></tr>\n{rows}</table>'
        )

    def _html_application_status(self, data: Dict[str, Any]) -> str:
        """Render HTML application status."""
        statuses = data.get("application_status", [])
        rows = ""
        for s in statuses:
            status = s.get("status", "Pending")
            cls = (
                "status-approved" if status == "Approved"
                else ("status-pending" if status == "Pending" else "status-rejected")
            )
            rows += (
                f'<tr><td>{s.get("program", "-")}</td>'
                f'<td>{s.get("measure", "-")}</td>'
                f'<td class="{cls}">{status}</td>'
                f'<td>{self._format_currency(s.get("amount", 0))}</td></tr>\n'
            )
        return (
            '<h2>Application Status</h2>\n'
            '<table>\n<tr><th>Program</th><th>Measure</th>'
            f'<th>Status</th><th>Amount</th></tr>\n{rows}</table>'
        )

    def _html_net_cost_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML net cost impact."""
        impact = data.get("net_cost_impact", {})
        return (
            '<h2>Net Cost Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Gross Investment</span>'
            f'<span class="value">{self._format_currency(impact.get("gross_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Total Rebates</span>'
            f'<span class="value rebate-value">{self._format_currency(impact.get("total_rebates", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Net Investment</span>'
            f'<span class="value">{self._format_currency(impact.get("net_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Effective Payback</span>'
            f'<span class="value">{self._fmt(impact.get("effective_payback_months", 0), 1)} mo</span></div>\n'
            '</div>'
        )

    def _html_stacking_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML stacking analysis."""
        stacking = data.get("stacking_analysis", [])
        rows = ""
        for s in stacking:
            programs = ", ".join(s.get("programs", []))
            rows += (
                f'<tr><td>{s.get("measure", "-")}</td>'
                f'<td>{programs}</td>'
                f'<td>{self._format_currency(s.get("combined_rebate", 0))}</td>'
                f'<td>{self._fmt(s.get("cost_coverage_pct", 0))}%</td>'
                f'<td>{"Yes" if s.get("stackable", False) else "No"}</td></tr>\n'
            )
        return (
            '<h2>Stacking Analysis</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Programs</th><th>Combined Rebate</th>'
            f'<th>Coverage</th><th>Stackable</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_rebate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON rebate summary."""
        s = data.get("rebate_summary", {})
        return {
            "programs_matched": s.get("programs_matched", 0),
            "total_potential_rebates": s.get("total_potential_rebates", 0),
            "rebates_applied": s.get("rebates_applied", 0),
            "rebates_received": s.get("rebates_received", 0),
            "rebates_pending": s.get("rebates_pending", 0),
            "coverage_rate_pct": s.get("coverage_rate_pct", 0),
            "net_investment_reduction_pct": s.get("net_investment_reduction_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        matches = data.get("program_matches", [])
        stacking = data.get("stacking_analysis", [])
        statuses = data.get("application_status", [])
        return {
            "rebate_by_program": {
                "type": "bar",
                "labels": [m.get("program", "") for m in matches],
                "values": [m.get("rebate_amount", 0) for m in matches],
            },
            "application_status_pie": {
                "type": "pie",
                "labels": list({s.get("status", "") for s in statuses}),
                "values": [
                    sum(1 for s in statuses if s.get("status") == status)
                    for status in {s.get("status", "") for s in statuses}
                ],
            },
            "stacking_bar": {
                "type": "stacked_bar",
                "labels": [s.get("measure", "") for s in stacking],
                "series": {
                    "combined_rebate": [s.get("combined_rebate", 0) for s in stacking],
                    "coverage_pct": [s.get("cost_coverage_pct", 0) for s in stacking],
                },
            },
            "cost_waterfall": {
                "type": "waterfall",
                "items": [
                    {
                        "label": "Gross Investment",
                        "value": data.get("net_cost_impact", {}).get("gross_investment", 0),
                    },
                    {
                        "label": "Rebates",
                        "value": -data.get("net_cost_impact", {}).get("total_rebates", 0),
                    },
                    {
                        "label": "Net Investment",
                        "value": data.get("net_cost_impact", {}).get("net_investment", 0),
                    },
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".rebate-value{color:#0d6efd;}"
            ".status-approved{color:#198754;font-weight:600;}"
            ".status-pending{color:#fd7e14;font-weight:600;}"
            ".status-rejected{color:#dc3545;font-weight:600;}"
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
