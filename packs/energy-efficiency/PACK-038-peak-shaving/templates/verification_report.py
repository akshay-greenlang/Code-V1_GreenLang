# -*- coding: utf-8 -*-
"""
VerificationReportTemplate - M&V verification for PACK-038.

Generates comprehensive measurement and verification (M&V) reports
for peak shaving programs showing baseline comparison with adjusted
baselines, savings verification using IPMVP-compliant methods,
regulatory compliance documentation, and performance verification
with statistical confidence intervals.

Sections:
    1. Verification Summary
    2. Baseline Comparison
    3. Savings Verification
    4. IPMVP Compliance
    5. Performance Documentation
    6. Statistical Analysis
    7. Certifications

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP (International Performance M&V Protocol)
    - ASHRAE Guideline 14 (measurement of energy and demand savings)
    - ISO 50015 (M&V of energy performance)
    - EVO IPMVP Core Concepts 2022

Author: GreenLang Team
Version: 38.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class VerificationReportTemplate:
    """
    Measurement and verification report template.

    Renders M&V verification reports showing baseline comparisons,
    savings verification, IPMVP compliance documentation, and
    performance statistics across markdown, HTML, and JSON formats.
    All outputs include SHA-256 provenance hashing for audit trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VerificationReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render verification report as Markdown.

        Args:
            data: Verification engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_verification_summary(data),
            self._md_baseline_comparison(data),
            self._md_savings_verification(data),
            self._md_ipmvp_compliance(data),
            self._md_performance_documentation(data),
            self._md_statistical_analysis(data),
            self._md_certifications(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render verification report as self-contained HTML.

        Args:
            data: Verification engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_verification_summary(data),
            self._html_baseline_comparison(data),
            self._html_savings_verification(data),
            self._html_ipmvp_compliance(data),
            self._html_performance_documentation(data),
            self._html_statistical_analysis(data),
            self._html_certifications(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Verification Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render verification report as structured JSON.

        Args:
            data: Verification engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "verification_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "verification_summary": self._json_verification_summary(data),
            "baseline_comparison": data.get("baseline_comparison", []),
            "savings_verification": self._json_savings_verification(data),
            "ipmvp_compliance": data.get("ipmvp_compliance", {}),
            "performance_documentation": data.get("performance_documentation", []),
            "statistical_analysis": self._json_statistical_analysis(data),
            "certifications": data.get("certifications", []),
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
            f"# Verification Report (M&V)\n\n"
            f"**Facility:** {facility}  \n"
            f"**Verification Period:** {data.get('verification_period', '')}  \n"
            f"**IPMVP Option:** {data.get('ipmvp_option', '')}  \n"
            f"**Verifier:** {data.get('verifier', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 VerificationReportTemplate v38.0.0\n\n---"
        )

    def _md_verification_summary(self, data: Dict[str, Any]) -> str:
        """Render verification summary section."""
        summary = data.get("verification_summary", {})
        return (
            "## 1. Verification Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Peak Demand | {self._format_power(summary.get('baseline_peak_kw', 0))} |\n"
            f"| Current Peak Demand | {self._format_power(summary.get('current_peak_kw', 0))} |\n"
            f"| Verified Peak Reduction | {self._format_power(summary.get('verified_reduction_kw', 0))} |\n"
            f"| Demand Savings Verified | {self._format_currency(summary.get('verified_savings', 0))} |\n"
            f"| Confidence Level | {self._fmt(summary.get('confidence_level_pct', 0))}% |\n"
            f"| Precision | +/-{self._fmt(summary.get('precision_pct', 0))}% |\n"
            f"| IPMVP Compliance | {summary.get('ipmvp_compliance_status', '-')} |\n"
            f"| Verification Result | {summary.get('verification_result', '-')} |"
        )

    def _md_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Render baseline comparison section."""
        comparisons = data.get("baseline_comparison", [])
        if not comparisons:
            return "## 2. Baseline Comparison\n\n_No baseline comparison data available._"
        lines = [
            "## 2. Baseline Comparison\n",
            "| Period | Baseline kW | Adjusted kW | Actual kW | Savings kW | Adj. Factor |",
            "|--------|----------:|----------:|--------:|----------:|----------:|",
        ]
        for comp in comparisons:
            lines.append(
                f"| {comp.get('period', '-')} "
                f"| {self._fmt(comp.get('baseline_kw', 0), 1)} "
                f"| {self._fmt(comp.get('adjusted_baseline_kw', 0), 1)} "
                f"| {self._fmt(comp.get('actual_kw', 0), 1)} "
                f"| {self._fmt(comp.get('savings_kw', 0), 1)} "
                f"| {self._fmt(comp.get('adjustment_factor', 0), 3)} |"
            )
        return "\n".join(lines)

    def _md_savings_verification(self, data: Dict[str, Any]) -> str:
        """Render savings verification section."""
        verification = data.get("savings_verification", {})
        if not verification:
            return "## 3. Savings Verification\n\n_No savings verification data available._"
        return (
            "## 3. Savings Verification\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Gross Demand Savings | {self._format_power(verification.get('gross_savings_kw', 0))} |\n"
            f"| Routine Adjustments | {self._format_power(verification.get('routine_adjustments_kw', 0))} |\n"
            f"| Non-Routine Adjustments | {self._format_power(verification.get('non_routine_adjustments_kw', 0))} |\n"
            f"| Net Verified Savings | {self._format_power(verification.get('net_savings_kw', 0))} |\n"
            f"| Savings as % of Baseline | {self._fmt(verification.get('savings_pct_baseline', 0))}% |\n"
            f"| Cost Savings (Verified) | {self._format_currency(verification.get('cost_savings', 0))} |\n"
            f"| Realization Rate | {self._fmt(verification.get('realization_rate_pct', 0))}% |\n"
            f"| Measurement Boundary | {verification.get('measurement_boundary', '-')} |"
        )

    def _md_ipmvp_compliance(self, data: Dict[str, Any]) -> str:
        """Render IPMVP compliance section."""
        compliance = data.get("ipmvp_compliance", {})
        if not compliance:
            return "## 4. IPMVP Compliance\n\n_No IPMVP compliance data available._"
        checklist = compliance.get("checklist", [])
        lines = [
            "## 4. IPMVP Compliance\n",
            f"**Option Selected:** {compliance.get('option', '-')}  \n"
            f"**Overall Status:** {compliance.get('overall_status', '-')}\n",
            "| Requirement | Status | Notes |",
            "|------------|--------|-------|",
        ]
        for item in checklist:
            status = "PASS" if item.get("compliant", False) else "FAIL"
            lines.append(
                f"| {item.get('requirement', '-')} "
                f"| {status} "
                f"| {item.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_performance_documentation(self, data: Dict[str, Any]) -> str:
        """Render performance documentation section."""
        docs = data.get("performance_documentation", [])
        if not docs:
            return "## 5. Performance Documentation\n\n_No performance documentation available._"
        lines = [
            "## 5. Performance Documentation\n",
            "| Event Date | Type | Baseline kW | Actual kW | Reduction | Duration | Result |",
            "|-----------|------|----------:|--------:|----------:|--------:|--------|",
        ]
        for doc in docs:
            lines.append(
                f"| {doc.get('event_date', '-')} "
                f"| {doc.get('event_type', '-')} "
                f"| {self._fmt(doc.get('baseline_kw', 0), 1)} "
                f"| {self._fmt(doc.get('actual_kw', 0), 1)} "
                f"| {self._fmt(doc.get('reduction_kw', 0), 1)} "
                f"| {doc.get('duration_min', 0)} min "
                f"| {doc.get('result', '-')} |"
            )
        return "\n".join(lines)

    def _md_statistical_analysis(self, data: Dict[str, Any]) -> str:
        """Render statistical analysis section."""
        stats = data.get("statistical_analysis", {})
        if not stats:
            return "## 6. Statistical Analysis\n\n_No statistical analysis available._"
        return (
            "## 6. Statistical Analysis\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Sample Size | {self._fmt(stats.get('sample_size', 0), 0)} |\n"
            f"| Mean Savings (kW) | {self._fmt(stats.get('mean_savings_kw', 0), 1)} |\n"
            f"| Std Deviation (kW) | {self._fmt(stats.get('std_dev_kw', 0), 1)} |\n"
            f"| t-Statistic | {self._fmt(stats.get('t_statistic', 0), 3)} |\n"
            f"| p-Value | {self._fmt(stats.get('p_value', 0), 4)} |\n"
            f"| Confidence Interval (90%) | {self._fmt(stats.get('ci_lower_kw', 0), 1)} - {self._fmt(stats.get('ci_upper_kw', 0), 1)} kW |\n"
            f"| R-squared (Baseline Model) | {self._fmt(stats.get('r_squared', 0), 4)} |\n"
            f"| CV-RMSE | {self._fmt(stats.get('cv_rmse_pct', 0))}% |"
        )

    def _md_certifications(self, data: Dict[str, Any]) -> str:
        """Render certifications section."""
        certs = data.get("certifications", [])
        if not certs:
            return (
                "## 7. Certifications\n\n"
                "This report has been prepared in accordance with IPMVP guidelines. "
                "The savings values presented have been independently verified."
            )
        lines = ["## 7. Certifications\n"]
        for cert in certs:
            lines.append(
                f"- **{cert.get('certifier', '-')}**: {cert.get('statement', '-')} "
                f"(Date: {cert.get('date', '-')})"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-038 Peak Shaving Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Verification Report (M&V)</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("verification_period", "-")} | '
            f'IPMVP: {data.get("ipmvp_option", "-")}</p>'
        )

    def _html_verification_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML verification summary cards."""
        s = data.get("verification_summary", {})
        result_color = "#198754" if s.get("verification_result") == "PASS" else "#dc3545"
        return (
            '<h2>Verification Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Verified Reduction</span>'
            f'<span class="value">{self._fmt(s.get("verified_reduction_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Verified Savings</span>'
            f'<span class="value">{self._format_currency(s.get("verified_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Confidence</span>'
            f'<span class="value">{self._fmt(s.get("confidence_level_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Precision</span>'
            f'<span class="value">+/-{self._fmt(s.get("precision_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Result</span>'
            f'<span class="value" style="color:{result_color}">'
            f'{s.get("verification_result", "-")}</span></div>\n'
            '</div>'
        )

    def _html_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline comparison table."""
        comparisons = data.get("baseline_comparison", [])
        rows = ""
        for comp in comparisons:
            rows += (
                f'<tr><td>{comp.get("period", "-")}</td>'
                f'<td>{self._fmt(comp.get("baseline_kw", 0), 1)}</td>'
                f'<td>{self._fmt(comp.get("adjusted_baseline_kw", 0), 1)}</td>'
                f'<td>{self._fmt(comp.get("actual_kw", 0), 1)}</td>'
                f'<td>{self._fmt(comp.get("savings_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Baseline Comparison</h2>\n'
            '<table>\n<tr><th>Period</th><th>Baseline kW</th><th>Adjusted kW</th>'
            f'<th>Actual kW</th><th>Savings kW</th></tr>\n{rows}</table>'
        )

    def _html_savings_verification(self, data: Dict[str, Any]) -> str:
        """Render HTML savings verification summary."""
        v = data.get("savings_verification", {})
        return (
            '<h2>Savings Verification</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Gross Savings</span>'
            f'<span class="value">{self._fmt(v.get("gross_savings_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Net Verified</span>'
            f'<span class="value">{self._fmt(v.get("net_savings_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(v.get("cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Realization Rate</span>'
            f'<span class="value">{self._fmt(v.get("realization_rate_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_ipmvp_compliance(self, data: Dict[str, Any]) -> str:
        """Render HTML IPMVP compliance table."""
        compliance = data.get("ipmvp_compliance", {})
        checklist = compliance.get("checklist", [])
        rows = ""
        for item in checklist:
            status = "PASS" if item.get("compliant", False) else "FAIL"
            color = "#198754" if status == "PASS" else "#dc3545"
            rows += (
                f'<tr><td>{item.get("requirement", "-")}</td>'
                f'<td style="color:{color};font-weight:700">{status}</td>'
                f'<td>{item.get("notes", "-")}</td></tr>\n'
            )
        return (
            '<h2>IPMVP Compliance</h2>\n'
            f'<p>Option: {compliance.get("option", "-")} | '
            f'Status: {compliance.get("overall_status", "-")}</p>\n'
            '<table>\n<tr><th>Requirement</th><th>Status</th>'
            f'<th>Notes</th></tr>\n{rows}</table>'
        )

    def _html_performance_documentation(self, data: Dict[str, Any]) -> str:
        """Render HTML performance documentation table."""
        docs = data.get("performance_documentation", [])
        rows = ""
        for doc in docs:
            rows += (
                f'<tr><td>{doc.get("event_date", "-")}</td>'
                f'<td>{doc.get("event_type", "-")}</td>'
                f'<td>{self._fmt(doc.get("baseline_kw", 0), 1)}</td>'
                f'<td>{self._fmt(doc.get("actual_kw", 0), 1)}</td>'
                f'<td>{self._fmt(doc.get("reduction_kw", 0), 1)}</td>'
                f'<td>{doc.get("result", "-")}</td></tr>\n'
            )
        return (
            '<h2>Performance Documentation</h2>\n'
            '<table>\n<tr><th>Date</th><th>Type</th><th>Baseline</th>'
            f'<th>Actual</th><th>Reduction</th><th>Result</th></tr>\n{rows}</table>'
        )

    def _html_statistical_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML statistical analysis summary."""
        s = data.get("statistical_analysis", {})
        return (
            '<h2>Statistical Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Mean Savings</span>'
            f'<span class="value">{self._fmt(s.get("mean_savings_kw", 0), 1)} kW</span></div>\n'
            f'  <div class="card"><span class="label">t-Statistic</span>'
            f'<span class="value">{self._fmt(s.get("t_statistic", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">p-Value</span>'
            f'<span class="value">{self._fmt(s.get("p_value", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value">{self._fmt(s.get("r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">CV-RMSE</span>'
            f'<span class="value">{self._fmt(s.get("cv_rmse_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_certifications(self, data: Dict[str, Any]) -> str:
        """Render HTML certifications."""
        certs = data.get("certifications", [])
        if not certs:
            return (
                '<h2>Certifications</h2>\n'
                '<p>This report has been prepared in accordance with IPMVP guidelines.</p>'
            )
        items = "".join(
            f'<li><strong>{c.get("certifier", "-")}:</strong> '
            f'{c.get("statement", "-")} (Date: {c.get("date", "-")})</li>\n'
            for c in certs
        )
        return f'<h2>Certifications</h2>\n<ul>\n{items}</ul>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_verification_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON verification summary."""
        s = data.get("verification_summary", {})
        return {
            "baseline_peak_kw": s.get("baseline_peak_kw", 0),
            "current_peak_kw": s.get("current_peak_kw", 0),
            "verified_reduction_kw": s.get("verified_reduction_kw", 0),
            "verified_savings": s.get("verified_savings", 0),
            "confidence_level_pct": s.get("confidence_level_pct", 0),
            "precision_pct": s.get("precision_pct", 0),
            "ipmvp_compliance_status": s.get("ipmvp_compliance_status", ""),
            "verification_result": s.get("verification_result", ""),
        }

    def _json_savings_verification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON savings verification."""
        v = data.get("savings_verification", {})
        return {
            "gross_savings_kw": v.get("gross_savings_kw", 0),
            "routine_adjustments_kw": v.get("routine_adjustments_kw", 0),
            "non_routine_adjustments_kw": v.get("non_routine_adjustments_kw", 0),
            "net_savings_kw": v.get("net_savings_kw", 0),
            "savings_pct_baseline": v.get("savings_pct_baseline", 0),
            "cost_savings": v.get("cost_savings", 0),
            "realization_rate_pct": v.get("realization_rate_pct", 0),
            "measurement_boundary": v.get("measurement_boundary", ""),
        }

    def _json_statistical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON statistical analysis."""
        s = data.get("statistical_analysis", {})
        return {
            "sample_size": s.get("sample_size", 0),
            "mean_savings_kw": s.get("mean_savings_kw", 0),
            "std_dev_kw": s.get("std_dev_kw", 0),
            "t_statistic": s.get("t_statistic", 0),
            "p_value": s.get("p_value", 0),
            "ci_lower_kw": s.get("ci_lower_kw", 0),
            "ci_upper_kw": s.get("ci_upper_kw", 0),
            "r_squared": s.get("r_squared", 0),
            "cv_rmse_pct": s.get("cv_rmse_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        comparisons = data.get("baseline_comparison", [])
        docs = data.get("performance_documentation", [])
        return {
            "baseline_vs_actual": {
                "type": "grouped_bar",
                "labels": [c.get("period", "") for c in comparisons],
                "series": {
                    "baseline": [c.get("adjusted_baseline_kw", 0) for c in comparisons],
                    "actual": [c.get("actual_kw", 0) for c in comparisons],
                },
            },
            "savings_trend": {
                "type": "line",
                "labels": [c.get("period", "") for c in comparisons],
                "values": [c.get("savings_kw", 0) for c in comparisons],
            },
            "event_performance": {
                "type": "scatter",
                "items": [
                    {
                        "date": d.get("event_date", ""),
                        "reduction_kw": d.get("reduction_kw", 0),
                        "result": d.get("result", ""),
                    }
                    for d in docs
                ],
            },
            "confidence_interval": {
                "type": "error_bar",
                "mean": data.get("statistical_analysis", {}).get("mean_savings_kw", 0),
                "lower": data.get("statistical_analysis", {}).get("ci_lower_kw", 0),
                "upper": data.get("statistical_analysis", {}).get("ci_upper_kw", 0),
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
            val: Energy value in kWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 kWh').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} kWh"
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
