# -*- coding: utf-8 -*-
"""
QuickWinsScanReportTemplate - Facility scan results report for PACK-033.

Generates a comprehensive report of quick-win opportunities identified
during a facility energy scan. Covers scan methodology, categorized
findings with savings estimates, payback periods, and priority rankings.

Sections:
    1. Executive Summary
    2. Facility Profile
    3. Scan Methodology
    4. Quick Wins Found (table)
    5. Category Breakdown
    6. Next Steps

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuickWinsScanReportTemplate:
    """
    Facility quick-wins scan report template.

    Renders scan results showing identified quick-win energy efficiency
    opportunities with savings estimates, payback analysis, and priority
    rankings across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize QuickWinsScanReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render quick-wins scan report as Markdown.

        Args:
            data: Scan engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_facility_profile(data),
            self._md_scan_methodology(data),
            self._md_quick_wins_found(data),
            self._md_category_breakdown(data),
            self._md_next_steps(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render quick-wins scan report as self-contained HTML.

        Args:
            data: Scan engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_facility_profile(data),
            self._html_quick_wins_found(data),
            self._html_category_breakdown(data),
            self._html_next_steps(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Quick Wins Scan Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render quick-wins scan report as structured JSON.

        Args:
            data: Scan engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "quick_wins_scan_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "facility_profile": data.get("facility_profile", {}),
            "scan_methodology": data.get("scan_methodology", {}),
            "quick_wins": data.get("quick_wins", []),
            "category_breakdown": self._json_category_breakdown(data),
            "next_steps": data.get("next_steps", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with scan metadata."""
        facility = data.get("facility_name", "Facility")
        scan_date = data.get("scan_date", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Quick Wins Scan Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Scan Date:** {scan_date}  \n"
            f"**Scan Type:** {data.get('scan_type', 'Walk-through Energy Scan')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 QuickWinsScanReportTemplate v33.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        summary = data.get("executive_summary", {})
        total_wins = summary.get("total_quick_wins", 0)
        total_savings = summary.get("total_annual_savings", 0)
        total_cost_savings = summary.get("total_cost_savings", 0)
        avg_payback = summary.get("average_payback_months", 0)
        total_co2 = summary.get("total_co2_reduction_tonnes", 0)
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Quick Wins Identified | {total_wins} |\n"
            f"| Total Annual Energy Savings | {self._format_energy(total_savings)} |\n"
            f"| Total Annual Cost Savings | {self._format_currency(total_cost_savings)} |\n"
            f"| Average Payback Period | {self._fmt(avg_payback, 1)} months |\n"
            f"| CO2 Reduction Potential | {self._fmt(total_co2)} tonnes/yr |\n"
            f"| Scan Coverage | {summary.get('scan_coverage_pct', 100)}% |"
        )

    def _md_facility_profile(self, data: Dict[str, Any]) -> str:
        """Render facility profile section."""
        profile = data.get("facility_profile", {})
        return (
            "## 2. Facility Profile\n\n"
            f"- **Name:** {profile.get('name', '-')}\n"
            f"- **Sector:** {profile.get('sector', '-')}\n"
            f"- **Floor Area:** {self._fmt(profile.get('floor_area_sqm', 0), 0)} sqm\n"
            f"- **Annual Energy Consumption:** {self._format_energy(profile.get('annual_energy_mwh', 0))}\n"
            f"- **Annual Energy Cost:** {self._format_currency(profile.get('annual_energy_cost', 0))}\n"
            f"- **Operating Hours:** {self._fmt(profile.get('operating_hours_yr', 0), 0)} hrs/yr\n"
            f"- **Energy Sources:** {', '.join(profile.get('energy_sources', ['-']))}"
        )

    def _md_scan_methodology(self, data: Dict[str, Any]) -> str:
        """Render scan methodology section."""
        method = data.get("scan_methodology", {})
        areas = method.get("areas_scanned", [])
        lines = [
            "## 3. Scan Methodology\n",
            f"- **Approach:** {method.get('approach', 'ASHRAE Level I Walk-through')}",
            f"- **Duration:** {method.get('duration_hours', 0)} hours",
            f"- **Team Size:** {method.get('team_size', 1)}",
            f"- **Equipment Used:** {', '.join(method.get('equipment', ['-']))}",
        ]
        if areas:
            lines.append("\n### Areas Scanned\n")
            for area in areas:
                lines.append(f"- {area}")
        return "\n".join(lines)

    def _md_quick_wins_found(self, data: Dict[str, Any]) -> str:
        """Render quick wins found table."""
        wins = data.get("quick_wins", [])
        if not wins:
            return "## 4. Quick Wins Found\n\n_No quick wins identified._"
        lines = [
            "## 4. Quick Wins Found\n",
            "| # | Action | Category | Annual Savings | Payback (mo) | Priority |",
            "|---|--------|----------|---------------|-------------|----------|",
        ]
        for i, w in enumerate(wins, 1):
            lines.append(
                f"| {i} | {w.get('action', '-')} "
                f"| {w.get('category', '-')} "
                f"| {self._format_currency(w.get('annual_cost_savings', 0))} "
                f"| {self._fmt(w.get('payback_months', 0), 1)} "
                f"| {w.get('priority', '-')} |"
            )
        total_savings = sum(w.get("annual_cost_savings", 0) for w in wins)
        lines.append(
            f"| | **TOTAL** | | **{self._format_currency(total_savings)}** | | |"
        )
        return "\n".join(lines)

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render category breakdown section with pie chart data."""
        breakdown = data.get("category_breakdown", [])
        if not breakdown:
            breakdown = self._compute_category_breakdown(data)
        if not breakdown:
            return "## 5. Category Breakdown\n\n_No breakdown data available._"
        lines = [
            "## 5. Category Breakdown\n",
            "| Category | Count | Savings Share (%) | Total Savings |",
            "|----------|-------|-------------------|---------------|",
        ]
        for cat in breakdown:
            lines.append(
                f"| {cat.get('category', '-')} "
                f"| {cat.get('count', 0)} "
                f"| {self._fmt(cat.get('savings_share_pct', 0))}% "
                f"| {self._format_currency(cat.get('total_savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        """Render next steps section."""
        steps = data.get("next_steps", [])
        if not steps:
            steps = [
                "Validate quick-win estimates with detailed engineering analysis",
                "Prioritize actions based on MCDA scoring",
                "Develop implementation timeline",
                "Identify rebate and incentive programs",
                "Assign implementation owners",
            ]
        lines = ["## 6. Next Steps\n"]
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
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
            f'<h1>Quick Wins Scan Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Scan Date: {data.get("scan_date", "-")}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Quick Wins</span>'
            f'<span class="value">{s.get("total_quick_wins", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._format_currency(s.get("total_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Avg Payback</span>'
            f'<span class="value">{self._fmt(s.get("average_payback_months", 0), 1)} mo</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Reduction</span>'
            f'<span class="value">{self._fmt(s.get("total_co2_reduction_tonnes", 0))} t/yr</span></div>\n'
            '</div>'
        )

    def _html_facility_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML facility profile."""
        p = data.get("facility_profile", {})
        return (
            '<h2>Facility Profile</h2>\n'
            f'<p><strong>{p.get("name", "-")}</strong> | '
            f'{p.get("sector", "-")} | '
            f'{self._fmt(p.get("floor_area_sqm", 0), 0)} sqm | '
            f'{self._format_energy(p.get("annual_energy_mwh", 0))}</p>'
        )

    def _html_quick_wins_found(self, data: Dict[str, Any]) -> str:
        """Render HTML quick wins table."""
        wins = data.get("quick_wins", [])
        rows = ""
        for w in wins:
            rows += (
                f'<tr><td>{w.get("action", "-")}</td>'
                f'<td>{w.get("category", "-")}</td>'
                f'<td>{self._format_currency(w.get("annual_cost_savings", 0))}</td>'
                f'<td>{self._fmt(w.get("payback_months", 0), 1)}</td>'
                f'<td><span class="priority-{w.get("priority", "medium").lower()}">'
                f'{w.get("priority", "-")}</span></td></tr>\n'
            )
        return (
            '<h2>Quick Wins Found</h2>\n'
            '<table>\n<tr><th>Action</th><th>Category</th>'
            f'<th>Annual Savings</th><th>Payback (mo)</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML category breakdown."""
        breakdown = data.get("category_breakdown", [])
        if not breakdown:
            breakdown = self._compute_category_breakdown(data)
        rows = ""
        for cat in breakdown:
            rows += (
                f'<tr><td>{cat.get("category", "-")}</td>'
                f'<td>{cat.get("count", 0)}</td>'
                f'<td>{self._fmt(cat.get("savings_share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Category Breakdown</h2>\n'
            '<table>\n<tr><th>Category</th><th>Count</th>'
            f'<th>Savings Share</th></tr>\n{rows}</table>'
        )

    def _html_next_steps(self, data: Dict[str, Any]) -> str:
        """Render HTML next steps."""
        steps = data.get("next_steps", [
            "Validate estimates with detailed analysis",
            "Prioritize using MCDA scoring",
            "Develop implementation timeline",
        ])
        items = "".join(f'<li>{s}</li>\n' for s in steps)
        return f'<h2>Next Steps</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        s = data.get("executive_summary", {})
        return {
            "total_quick_wins": s.get("total_quick_wins", 0),
            "total_annual_savings_mwh": s.get("total_annual_savings", 0),
            "total_cost_savings": s.get("total_cost_savings", 0),
            "average_payback_months": s.get("average_payback_months", 0),
            "total_co2_reduction_tonnes": s.get("total_co2_reduction_tonnes", 0),
            "scan_coverage_pct": s.get("scan_coverage_pct", 100),
        }

    def _json_category_breakdown(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON category breakdown."""
        breakdown = data.get("category_breakdown", [])
        if not breakdown:
            breakdown = self._compute_category_breakdown(data)
        return breakdown

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        wins = data.get("quick_wins", [])
        breakdown = data.get("category_breakdown", [])
        if not breakdown:
            breakdown = self._compute_category_breakdown(data)
        return {
            "category_pie": {
                "type": "pie",
                "labels": [c.get("category", "") for c in breakdown],
                "values": [c.get("total_savings", 0) for c in breakdown],
            },
            "priority_bar": {
                "type": "bar",
                "labels": ["High", "Medium", "Low"],
                "values": [
                    sum(1 for w in wins if w.get("priority", "").lower() == "high"),
                    sum(1 for w in wins if w.get("priority", "").lower() == "medium"),
                    sum(1 for w in wins if w.get("priority", "").lower() == "low"),
                ],
            },
            "payback_scatter": {
                "type": "scatter",
                "points": [
                    {
                        "x": w.get("payback_months", 0),
                        "y": w.get("annual_cost_savings", 0),
                        "label": w.get("action", ""),
                    }
                    for w in wins
                ],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_category_breakdown(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compute category breakdown from quick wins list."""
        wins = data.get("quick_wins", [])
        if not wins:
            return []
        categories: Dict[str, Dict[str, Any]] = {}
        total_savings = sum(w.get("annual_cost_savings", 0) for w in wins)
        for w in wins:
            cat = w.get("category", "Other")
            if cat not in categories:
                categories[cat] = {"category": cat, "count": 0, "total_savings": 0}
            categories[cat]["count"] += 1
            categories[cat]["total_savings"] += w.get("annual_cost_savings", 0)
        for cat_data in categories.values():
            if total_savings > 0:
                cat_data["savings_share_pct"] = round(
                    (cat_data["total_savings"] / total_savings) * 100, 1
                )
            else:
                cat_data["savings_share_pct"] = 0.0
        return sorted(categories.values(), key=lambda x: x["total_savings"], reverse=True)

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
            ".priority-high{color:#dc3545;font-weight:700;}"
            ".priority-medium{color:#fd7e14;font-weight:600;}"
            ".priority-low{color:#198754;font-weight:500;}"
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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
