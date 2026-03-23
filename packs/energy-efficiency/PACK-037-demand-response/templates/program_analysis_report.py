# -*- coding: utf-8 -*-
"""
ProgramAnalysisReportTemplate - DR program comparison for PACK-037.

Generates demand response program analysis reports comparing available DR
programs by eligibility requirements, projected revenue, penalty structures,
risk profiles, and enrollment recommendations. Covers utility, ISO/RTO,
and aggregator programs with side-by-side comparison matrices.

Sections:
    1. Program Overview
    2. Eligibility Assessment
    3. Revenue Projections
    4. Penalty & Risk Analysis
    5. Program Comparison Matrix
    6. Enrollment Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 745 (compensation for DR)
    - FERC Order 2222 (DER participation)
    - EU Clean Energy Package (demand response provisions)

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class ProgramAnalysisReportTemplate:
    """
    DR program comparison and analysis report template.

    Renders demand response program comparisons with eligibility checks,
    revenue projections, penalty risk analysis, and enrollment
    recommendations across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgramAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render program analysis report as Markdown.

        Args:
            data: Program analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_program_overview(data),
            self._md_eligibility_assessment(data),
            self._md_revenue_projections(data),
            self._md_penalty_risk_analysis(data),
            self._md_comparison_matrix(data),
            self._md_enrollment_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render program analysis report as self-contained HTML.

        Args:
            data: Program analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_program_overview(data),
            self._html_eligibility_assessment(data),
            self._html_revenue_projections(data),
            self._html_penalty_risk_analysis(data),
            self._html_comparison_matrix(data),
            self._html_enrollment_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DR Program Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render program analysis report as structured JSON.

        Args:
            data: Program analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "program_analysis_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "program_overview": self._json_program_overview(data),
            "eligibility_assessment": data.get("eligibility_assessment", []),
            "revenue_projections": data.get("revenue_projections", []),
            "penalty_risk_analysis": data.get("penalty_risk_analysis", []),
            "comparison_matrix": data.get("comparison_matrix", []),
            "enrollment_recommendations": data.get("enrollment_recommendations", []),
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
            f"# Demand Response Program Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Date:** {data.get('analysis_date', '')}  \n"
            f"**Utility/ISO:** {data.get('utility_iso', '')}  \n"
            f"**Curtailable Capacity:** {self._format_power(data.get('curtailable_capacity_kw', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 ProgramAnalysisReportTemplate v37.0.0\n\n---"
        )

    def _md_program_overview(self, data: Dict[str, Any]) -> str:
        """Render program overview section."""
        overview = data.get("program_overview", {})
        programs = data.get("programs", [])
        return (
            "## 1. Program Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Programs Analyzed | {overview.get('programs_analyzed', len(programs))} |\n"
            f"| Eligible Programs | {overview.get('eligible_count', 0)} |\n"
            f"| Max Annual Revenue Potential | {self._format_currency(overview.get('max_annual_revenue', 0))} |\n"
            f"| Recommended Programs | {overview.get('recommended_count', 0)} |\n"
            f"| Program Season | {overview.get('program_season', 'Year-round')} |\n"
            f"| Commitment Period | {overview.get('commitment_period', '-')} |"
        )

    def _md_eligibility_assessment(self, data: Dict[str, Any]) -> str:
        """Render eligibility assessment section."""
        programs = data.get("eligibility_assessment", [])
        if not programs:
            return "## 2. Eligibility Assessment\n\n_No programs assessed._"
        lines = [
            "## 2. Eligibility Assessment\n",
            "| Program | Min kW | Notification | Duration | Eligible | Gap |",
            "|---------|-------:|------------:|--------:|----------|-----|",
        ]
        for p in programs:
            status = "Yes" if p.get("eligible", False) else "No"
            lines.append(
                f"| {p.get('program_name', '-')} "
                f"| {self._fmt(p.get('min_capacity_kw', 0), 0)} "
                f"| {p.get('notification_required', '-')} "
                f"| {p.get('min_duration', '-')} "
                f"| {status} "
                f"| {p.get('gap_description', '-')} |"
            )
        return "\n".join(lines)

    def _md_revenue_projections(self, data: Dict[str, Any]) -> str:
        """Render revenue projections section."""
        projections = data.get("revenue_projections", [])
        if not projections:
            return "## 3. Revenue Projections\n\n_No revenue data available._"
        lines = [
            "## 3. Revenue Projections\n",
            "| Program | Capacity Payment | Energy Payment | Ancillary | Total Annual | Confidence |",
            "|---------|----------------:|---------------:|----------:|------------:|-----------|",
        ]
        for p in projections:
            total = (
                p.get("capacity_payment", 0)
                + p.get("energy_payment", 0)
                + p.get("ancillary_payment", 0)
            )
            lines.append(
                f"| {p.get('program_name', '-')} "
                f"| {self._format_currency(p.get('capacity_payment', 0))} "
                f"| {self._format_currency(p.get('energy_payment', 0))} "
                f"| {self._format_currency(p.get('ancillary_payment', 0))} "
                f"| {self._format_currency(total)} "
                f"| {p.get('confidence', '-')} |"
            )
        grand_total = sum(
            p.get("capacity_payment", 0) + p.get("energy_payment", 0) + p.get("ancillary_payment", 0)
            for p in projections
        )
        lines.append(
            f"| **TOTAL** | | | | **{self._format_currency(grand_total)}** | |"
        )
        return "\n".join(lines)

    def _md_penalty_risk_analysis(self, data: Dict[str, Any]) -> str:
        """Render penalty and risk analysis section."""
        risks = data.get("penalty_risk_analysis", [])
        if not risks:
            return "## 4. Penalty & Risk Analysis\n\n_No penalty data available._"
        lines = [
            "## 4. Penalty & Risk Analysis\n",
            "| Program | Non-Performance Penalty | Max Annual Exposure | Strike Count | Risk Level |",
            "|---------|----------------------:|-------------------:|------------:|-----------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('program_name', '-')} "
                f"| {self._format_currency(r.get('non_performance_penalty', 0))} "
                f"| {self._format_currency(r.get('max_annual_exposure', 0))} "
                f"| {r.get('strike_count_limit', '-')} "
                f"| {r.get('risk_level', '-')} |"
            )
        return "\n".join(lines)

    def _md_comparison_matrix(self, data: Dict[str, Any]) -> str:
        """Render program comparison matrix section."""
        matrix = data.get("comparison_matrix", [])
        if not matrix:
            return "## 5. Program Comparison Matrix\n\n_No comparison data._"
        lines = [
            "## 5. Program Comparison Matrix\n",
            "| Criterion | " + " | ".join(p.get("program_name", "-") for p in matrix) + " |",
            "|-----------|" + "|".join(["-----" for _ in matrix]) + "|",
        ]
        criteria = [
            ("Revenue Potential", "revenue_score"),
            ("Risk Level", "risk_score"),
            ("Operational Fit", "operational_fit_score"),
            ("Flexibility Match", "flexibility_match_score"),
            ("Overall Score", "overall_score"),
        ]
        for label, key in criteria:
            row = f"| {label} |"
            for p in matrix:
                row += f" {self._fmt(p.get(key, 0), 1)} |"
            lines.append(row)
        return "\n".join(lines)

    def _md_enrollment_recommendations(self, data: Dict[str, Any]) -> str:
        """Render enrollment recommendations section."""
        recs = data.get("enrollment_recommendations", [])
        if not recs:
            recs = [
                {"program": "Primary DR program", "action": "Enroll immediately",
                 "rationale": "Highest revenue-to-risk ratio"},
            ]
        lines = [
            "## 6. Enrollment Recommendations\n",
            "| # | Program | Action | Rationale |",
            "|---|---------|--------|-----------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('program', '-')} "
                f"| {r.get('action', '-')} "
                f"| {r.get('rationale', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>DR Program Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Utility/ISO: {data.get("utility_iso", "-")} | '
            f'Capacity: {self._format_power(data.get("curtailable_capacity_kw", 0))}</p>'
        )

    def _html_program_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML program overview cards."""
        o = data.get("program_overview", {})
        return (
            '<h2>Program Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Programs Analyzed</span>'
            f'<span class="value">{o.get("programs_analyzed", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Eligible</span>'
            f'<span class="value">{o.get("eligible_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Max Revenue</span>'
            f'<span class="value">{self._format_currency(o.get("max_annual_revenue", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Recommended</span>'
            f'<span class="value">{o.get("recommended_count", 0)}</span></div>\n'
            '</div>'
        )

    def _html_eligibility_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML eligibility assessment table."""
        programs = data.get("eligibility_assessment", [])
        rows = ""
        for p in programs:
            cls = "status-pass" if p.get("eligible", False) else "status-fail"
            rows += (
                f'<tr><td>{p.get("program_name", "-")}</td>'
                f'<td>{self._fmt(p.get("min_capacity_kw", 0), 0)}</td>'
                f'<td>{p.get("notification_required", "-")}</td>'
                f'<td class="{cls}">{"Eligible" if p.get("eligible", False) else "Not Eligible"}</td></tr>\n'
            )
        return (
            '<h2>Eligibility Assessment</h2>\n'
            '<table>\n<tr><th>Program</th><th>Min kW</th>'
            f'<th>Notification</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_revenue_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue projections table."""
        projections = data.get("revenue_projections", [])
        rows = ""
        for p in projections:
            total = (
                p.get("capacity_payment", 0) + p.get("energy_payment", 0)
                + p.get("ancillary_payment", 0)
            )
            rows += (
                f'<tr><td>{p.get("program_name", "-")}</td>'
                f'<td>{self._format_currency(p.get("capacity_payment", 0))}</td>'
                f'<td>{self._format_currency(p.get("energy_payment", 0))}</td>'
                f'<td>{self._format_currency(total)}</td></tr>\n'
            )
        return (
            '<h2>Revenue Projections</h2>\n'
            '<table>\n<tr><th>Program</th><th>Capacity</th>'
            f'<th>Energy</th><th>Total Annual</th></tr>\n{rows}</table>'
        )

    def _html_penalty_risk_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML penalty risk analysis."""
        risks = data.get("penalty_risk_analysis", [])
        rows = ""
        for r in risks:
            risk_cls = f'risk-{r.get("risk_level", "low").lower()}'
            rows += (
                f'<tr><td>{r.get("program_name", "-")}</td>'
                f'<td>{self._format_currency(r.get("non_performance_penalty", 0))}</td>'
                f'<td>{self._format_currency(r.get("max_annual_exposure", 0))}</td>'
                f'<td class="{risk_cls}">{r.get("risk_level", "-")}</td></tr>\n'
            )
        return (
            '<h2>Penalty &amp; Risk Analysis</h2>\n'
            '<table>\n<tr><th>Program</th><th>Penalty</th>'
            f'<th>Max Exposure</th><th>Risk</th></tr>\n{rows}</table>'
        )

    def _html_comparison_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML comparison matrix."""
        matrix = data.get("comparison_matrix", [])
        if not matrix:
            return '<h2>Program Comparison</h2>\n<p>No comparison data available.</p>'
        headers = "".join(f'<th>{p.get("program_name", "-")}</th>' for p in matrix)
        criteria = [
            ("Revenue", "revenue_score"),
            ("Risk", "risk_score"),
            ("Fit", "operational_fit_score"),
            ("Overall", "overall_score"),
        ]
        rows = ""
        for label, key in criteria:
            cells = "".join(f'<td>{self._fmt(p.get(key, 0), 1)}</td>' for p in matrix)
            rows += f'<tr><td><strong>{label}</strong></td>{cells}</tr>\n'
        return (
            '<h2>Program Comparison</h2>\n'
            f'<table>\n<tr><th>Criterion</th>{headers}</tr>\n{rows}</table>'
        )

    def _html_enrollment_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML enrollment recommendations."""
        recs = data.get("enrollment_recommendations", [])
        items = "".join(
            f'<li><strong>{r.get("program", "-")}</strong>: {r.get("action", "-")} '
            f'- {r.get("rationale", "-")}</li>\n'
            for r in recs
        )
        return f'<h2>Enrollment Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_program_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON program overview."""
        o = data.get("program_overview", {})
        return {
            "programs_analyzed": o.get("programs_analyzed", 0),
            "eligible_count": o.get("eligible_count", 0),
            "max_annual_revenue": o.get("max_annual_revenue", 0),
            "recommended_count": o.get("recommended_count", 0),
            "program_season": o.get("program_season", "Year-round"),
            "commitment_period": o.get("commitment_period", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        projections = data.get("revenue_projections", [])
        risks = data.get("penalty_risk_analysis", [])
        matrix = data.get("comparison_matrix", [])
        return {
            "revenue_bar": {
                "type": "bar",
                "labels": [p.get("program_name", "") for p in projections],
                "series": {
                    "capacity": [p.get("capacity_payment", 0) for p in projections],
                    "energy": [p.get("energy_payment", 0) for p in projections],
                    "ancillary": [p.get("ancillary_payment", 0) for p in projections],
                },
            },
            "risk_exposure_bar": {
                "type": "bar",
                "labels": [r.get("program_name", "") for r in risks],
                "values": [r.get("max_annual_exposure", 0) for r in risks],
            },
            "comparison_radar": {
                "type": "radar",
                "labels": ["Revenue", "Risk", "Operational Fit", "Flexibility Match"],
                "series": {
                    p.get("program_name", f"Program {i}"): [
                        p.get("revenue_score", 0),
                        p.get("risk_score", 0),
                        p.get("operational_fit_score", 0),
                        p.get("flexibility_match_score", 0),
                    ]
                    for i, p in enumerate(matrix)
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
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".risk-high{color:#dc3545;font-weight:700;}"
            ".risk-medium{color:#fd7e14;font-weight:600;}"
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

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
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
