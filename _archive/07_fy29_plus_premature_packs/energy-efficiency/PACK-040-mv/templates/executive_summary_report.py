# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - Executive Summary Report for PACK-040.

Generates concise 2-4 page executive summary reports covering verified
savings highlights, financial impact assessment, compliance status
overview, risk indicators, performance trends, and strategic
recommendations for senior leadership and stakeholders.

Sections:
    1. Executive Overview
    2. Verified Savings
    3. Financial Impact
    4. Compliance Status
    5. Performance Trends
    6. Risk Assessment
    7. ECM Portfolio
    8. Outlook
    9. Action Items
    10. Appendix References

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (executive reporting)
    - ISO 50015:2014 (management reporting)
    - FEMP M&V Guidelines 4.0 (reporting)

Author: GreenLang Team
Version: 40.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template.

    Renders concise 2-4 page executive summary reports showing verified
    savings highlights, financial impact with ROI metrics, compliance
    status traffic lights, performance trend indicators, risk
    assessment, ECM portfolio overview, outlook projections, and
    prioritized action items across markdown, HTML, and JSON formats.
    All outputs include SHA-256 provenance hashing for audit trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary report as Markdown.

        Args:
            data: M&V reporting engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_overview(data),
            self._md_verified_savings(data),
            self._md_financial_impact(data),
            self._md_compliance_status(data),
            self._md_performance_trends(data),
            self._md_risk_assessment(data),
            self._md_ecm_portfolio(data),
            self._md_outlook(data),
            self._md_action_items(data),
            self._md_appendix_references(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary report as self-contained HTML.

        Args:
            data: M&V reporting engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_overview(data),
            self._html_verified_savings(data),
            self._html_financial_impact(data),
            self._html_compliance_status(data),
            self._html_performance_trends(data),
            self._html_risk_assessment(data),
            self._html_ecm_portfolio(data),
            self._html_outlook(data),
            self._html_action_items(data),
            self._html_appendix_references(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>M&amp;V Executive Summary</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary report as structured JSON.

        Args:
            data: M&V reporting engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_summary_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_overview": self._json_overview(data),
            "verified_savings": data.get("verified_savings", {}),
            "financial_impact": data.get("financial_impact", {}),
            "compliance_status": data.get("compliance_status", {}),
            "performance_trends": data.get("performance_trends", {}),
            "risk_assessment": data.get("risk_assessment", []),
            "ecm_portfolio": data.get("ecm_portfolio", []),
            "outlook": data.get("outlook", {}),
            "action_items": data.get("action_items", []),
            "appendix_references": data.get("appendix_references", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with project metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# M&V Executive Summary\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Prepared For:** {data.get('prepared_for', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 ExecutiveSummaryReportTemplate v40.0.0\n\n---"
        )

    def _md_executive_overview(self, data: Dict[str, Any]) -> str:
        """Render executive overview section."""
        o = data.get("executive_overview", {})
        return (
            "## 1. Executive Overview\n\n"
            f"{o.get('narrative', 'No executive narrative provided.')}\n\n"
            "| Key Metric | Value |\n|------------|-------|\n"
            f"| Total Verified Savings | {self._format_energy(o.get('total_savings_mwh', 0))} |\n"
            f"| Total Cost Savings | {self._format_currency(o.get('total_cost_savings', 0))} |\n"
            f"| Target Achievement | {self._fmt(o.get('target_achievement_pct', 0))}% |\n"
            f"| CO2 Avoided | {self._fmt(o.get('co2_avoided_tonnes', 0), 1)} tCO2e |\n"
            f"| Confidence Level | {self._fmt(o.get('confidence_level_pct', 90))}% |\n"
            f"| Overall Status | {o.get('overall_status', '-')} |"
        )

    def _md_verified_savings(self, data: Dict[str, Any]) -> str:
        """Render verified savings section."""
        vs = data.get("verified_savings", {})
        if not vs:
            return "## 2. Verified Savings\n\n_No verified savings data available._"
        return (
            "## 2. Verified Savings\n\n"
            "| Component | Value |\n|-----------|-------|\n"
            f"| Adjusted Baseline | {self._format_energy(vs.get('adjusted_baseline_mwh', 0))} |\n"
            f"| Actual Consumption | {self._format_energy(vs.get('actual_consumption_mwh', 0))} |\n"
            f"| Avoided Energy | {self._format_energy(vs.get('avoided_energy_mwh', 0))} |\n"
            f"| Savings Percentage | {self._fmt(vs.get('savings_pct', 0))}% |\n"
            f"| Uncertainty (+/-) | {self._format_energy(vs.get('uncertainty_mwh', 0))} |\n"
            f"| Statistically Significant | {vs.get('is_significant', '-')} |\n"
            f"| Realization Rate | {self._fmt(vs.get('realization_rate_pct', 0))}% |"
        )

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render financial impact section."""
        fi = data.get("financial_impact", {})
        if not fi:
            return "## 3. Financial Impact\n\n_No financial impact data available._"
        return (
            "## 3. Financial Impact\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Annual Cost Savings | {self._format_currency(fi.get('annual_cost_savings', 0))} |\n"
            f"| Cumulative Cost Savings | {self._format_currency(fi.get('cumulative_cost_savings', 0))} |\n"
            f"| Project Investment | {self._format_currency(fi.get('project_investment', 0))} |\n"
            f"| Simple Payback | {self._fmt(fi.get('simple_payback_years', 0), 1)} years |\n"
            f"| ROI | {self._fmt(fi.get('roi_pct', 0))}% |\n"
            f"| NPV | {self._format_currency(fi.get('npv', 0))} |\n"
            f"| IRR | {self._fmt(fi.get('irr_pct', 0))}% |\n"
            f"| M&V Cost | {self._format_currency(fi.get('mv_cost', 0))} |\n"
            f"| M&V / Savings Ratio | {self._fmt(fi.get('mv_savings_ratio_pct', 0))}% |"
        )

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render compliance status section."""
        comp = data.get("compliance_status", {})
        if not comp:
            return "## 4. Compliance Status\n\n_No compliance status data available._"
        items = comp.get("items", [])
        lines = [
            "## 4. Compliance Status\n",
            f"**Overall:** {comp.get('overall', '-')}  \n",
        ]
        if items:
            lines.append("| Standard | Status | Notes |")
            lines.append("|----------|:------:|-------|")
            for item in items:
                lines.append(
                    f"| {item.get('standard', '-')} "
                    f"| {item.get('status', '-')} "
                    f"| {item.get('notes', '-')} |"
                )
        return "\n".join(lines)

    def _md_performance_trends(self, data: Dict[str, Any]) -> str:
        """Render performance trends section."""
        trends = data.get("performance_trends", {})
        if not trends:
            return "## 5. Performance Trends\n\n_No performance trend data available._"
        return (
            "## 5. Performance Trends\n\n"
            "| Indicator | Value | Trend | Status |\n|-----------|------:|:-----:|:------:|\n"
            f"| Savings Rate | {self._fmt(trends.get('savings_rate_pct', 0))}% | {trends.get('savings_trend', '-')} | {trends.get('savings_status', '-')} |\n"
            f"| Persistence Factor | {self._fmt(trends.get('persistence_factor', 0), 3)} | {trends.get('persistence_trend', '-')} | {trends.get('persistence_status', '-')} |\n"
            f"| Model Accuracy | {self._fmt(trends.get('model_accuracy_pct', 0))}% | {trends.get('accuracy_trend', '-')} | {trends.get('accuracy_status', '-')} |\n"
            f"| Data Quality | {self._fmt(trends.get('data_quality_pct', 0))}% | {trends.get('quality_trend', '-')} | {trends.get('quality_status', '-')} |"
        )

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        risks = data.get("risk_assessment", [])
        if not risks:
            return "## 6. Risk Assessment\n\n_No risk data available._"
        lines = [
            "## 6. Risk Assessment\n",
            "| Risk | Severity | Likelihood | Mitigation |",
            "|------|:--------:|:----------:|------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('severity', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_ecm_portfolio(self, data: Dict[str, Any]) -> str:
        """Render ECM portfolio section."""
        ecms = data.get("ecm_portfolio", [])
        if not ecms:
            return "## 7. ECM Portfolio\n\n_No ECM portfolio data available._"
        lines = [
            "## 7. ECM Portfolio\n",
            "| ECM | Verified (MWh) | Cost Savings | Realization (%) | Status |",
            "|-----|-------------:|-------------:|:--------------:|--------|",
        ]
        for ecm in ecms:
            lines.append(
                f"| {ecm.get('name', '-')} "
                f"| {self._fmt(ecm.get('verified_mwh', 0), 1)} "
                f"| {self._format_currency(ecm.get('cost_savings', 0))} "
                f"| {self._fmt(ecm.get('realization_pct', 0))}% "
                f"| {ecm.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_outlook(self, data: Dict[str, Any]) -> str:
        """Render outlook section."""
        outlook = data.get("outlook", {})
        if not outlook:
            return "## 8. Outlook\n\n_No outlook data available._"
        return (
            "## 8. Outlook\n\n"
            f"{outlook.get('narrative', 'No outlook narrative provided.')}\n\n"
            "| Projection | Value |\n|------------|-------|\n"
            f"| Next Year Savings | {self._format_energy(outlook.get('next_year_savings_mwh', 0))} |\n"
            f"| Next Year Cost Savings | {self._format_currency(outlook.get('next_year_cost_savings', 0))} |\n"
            f"| 5-Year Cumulative | {self._format_energy(outlook.get('five_year_cumulative_mwh', 0))} |\n"
            f"| Persistence Projection | {self._fmt(outlook.get('persistence_projection', 0), 3)} |\n"
            f"| Key Opportunities | {outlook.get('key_opportunities', '-')} |"
        )

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items section."""
        actions = data.get("action_items", [])
        if not actions:
            return "## 9. Action Items\n\n_No action items available._"
        lines = ["## 9. Action Items\n"]
        for i, action in enumerate(actions, 1):
            lines.append(
                f"{i}. **[{action.get('priority', 'Medium')}]** "
                f"{action.get('description', '-')} "
                f"(Owner: {action.get('owner', '-')}, Due: {action.get('due_date', '-')})"
            )
        return "\n".join(lines)

    def _md_appendix_references(self, data: Dict[str, Any]) -> str:
        """Render appendix references section."""
        refs = data.get("appendix_references", [])
        if not refs:
            return "## 10. Appendix References\n\n_See detailed reports for full analysis._"
        lines = ["## 10. Appendix References\n"]
        for ref in refs:
            lines.append(f"- **{ref.get('name', '-')}:** {ref.get('description', '-')}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-040 M&V Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>M&amp;V Executive Summary</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'For: {data.get("prepared_for", "-")}</p>'
        )

    def _html_executive_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML executive overview cards."""
        o = data.get("executive_overview", {})
        status_cls = "severity-low" if o.get("overall_status") == "On Track" else "severity-medium"
        return (
            '<h2>1. Executive Overview</h2>\n'
            f'<p>{o.get("narrative", "No executive narrative provided.")}</p>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Savings</span>'
            f'<span class="value">{self._fmt(o.get("total_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(o.get("total_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Target</span>'
            f'<span class="value">{self._fmt(o.get("target_achievement_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Avoided</span>'
            f'<span class="value">{self._fmt(o.get("co2_avoided_tonnes", 0), 1)} t</span></div>\n'
            f'  <div class="card"><span class="label">Status</span>'
            f'<span class="value {status_cls}">{o.get("overall_status", "-")}</span></div>\n'
            '</div>'
        )

    def _html_verified_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML verified savings table."""
        vs = data.get("verified_savings", {})
        sig_cls = "severity-low" if vs.get("is_significant") else "severity-high"
        return (
            '<h2>2. Verified Savings</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Avoided Energy</span>'
            f'<span class="value">{self._fmt(vs.get("avoided_energy_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Savings %</span>'
            f'<span class="value">{self._fmt(vs.get("savings_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Realization</span>'
            f'<span class="value">{self._fmt(vs.get("realization_rate_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Significant</span>'
            f'<span class="value {sig_cls}">{vs.get("is_significant", "-")}</span></div>\n'
            '</div>'
        )

    def _html_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML financial impact cards."""
        fi = data.get("financial_impact", {})
        return (
            '<h2>3. Financial Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._format_currency(fi.get("annual_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(fi.get("simple_payback_years", 0), 1)} yrs</span></div>\n'
            f'  <div class="card"><span class="label">ROI</span>'
            f'<span class="value">{self._fmt(fi.get("roi_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(fi.get("npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(fi.get("irr_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status table."""
        comp = data.get("compliance_status", {})
        items = comp.get("items", [])
        rows = ""
        for item in items:
            cls = "severity-low" if item.get("status") == "PASS" else "severity-high"
            rows += (
                f'<tr><td>{item.get("standard", "-")}</td>'
                f'<td class="{cls}">{item.get("status", "-")}</td>'
                f'<td>{item.get("notes", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Compliance Status</h2>\n'
            f'<p>Overall: {comp.get("overall", "-")}</p>\n'
            '<table>\n<tr><th>Standard</th><th>Status</th>'
            f'<th>Notes</th></tr>\n{rows}</table>'
        )

    def _html_performance_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML performance trends cards."""
        trends = data.get("performance_trends", {})
        return (
            '<h2>5. Performance Trends</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Savings Rate</span>'
            f'<span class="value">{self._fmt(trends.get("savings_rate_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Persistence</span>'
            f'<span class="value">{self._fmt(trends.get("persistence_factor", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Model Accuracy</span>'
            f'<span class="value">{self._fmt(trends.get("model_accuracy_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Data Quality</span>'
            f'<span class="value">{self._fmt(trends.get("data_quality_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment table."""
        risks = data.get("risk_assessment", [])
        rows = ""
        for r in risks:
            cls = "severity-high" if r.get("severity") == "High" else (
                "severity-medium" if r.get("severity") == "Medium" else "severity-low"
            )
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td class="{cls}">{r.get("severity", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Risk Assessment</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Severity</th>'
            f'<th>Likelihood</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_ecm_portfolio(self, data: Dict[str, Any]) -> str:
        """Render HTML ECM portfolio table."""
        ecms = data.get("ecm_portfolio", [])
        rows = ""
        for ecm in ecms:
            cls = "severity-low" if ecm.get("realization_pct", 0) >= 80 else "severity-high"
            rows += (
                f'<tr><td>{ecm.get("name", "-")}</td>'
                f'<td>{self._fmt(ecm.get("verified_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(ecm.get("cost_savings", 0))}</td>'
                f'<td class="{cls}">{self._fmt(ecm.get("realization_pct", 0))}%</td>'
                f'<td>{ecm.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>7. ECM Portfolio</h2>\n'
            '<table>\n<tr><th>ECM</th><th>Verified (MWh)</th><th>Cost Savings</th>'
            f'<th>Realization</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_outlook(self, data: Dict[str, Any]) -> str:
        """Render HTML outlook section."""
        outlook = data.get("outlook", {})
        return (
            '<h2>8. Outlook</h2>\n'
            f'<p>{outlook.get("narrative", "No outlook narrative provided.")}</p>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Next Year</span>'
            f'<span class="value">{self._fmt(outlook.get("next_year_savings_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Next Year Cost</span>'
            f'<span class="value">{self._format_currency(outlook.get("next_year_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">5-Year Total</span>'
            f'<span class="value">{self._fmt(outlook.get("five_year_cumulative_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Persistence</span>'
            f'<span class="value">{self._fmt(outlook.get("persistence_projection", 0), 3)}</span></div>\n'
            '</div>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        actions = data.get("action_items", [])
        items = ""
        for action in actions:
            pri = action.get("priority", "Medium")
            cls = "severity-high" if pri == "High" else (
                "severity-medium" if pri == "Medium" else "severity-low"
            )
            items += (
                f'<li><span class="{cls}">[{pri}]</span> '
                f'{action.get("description", "-")} '
                f'(Owner: {action.get("owner", "-")}, Due: {action.get("due_date", "-")})</li>\n'
            )
        return f'<h2>9. Action Items</h2>\n<ol>\n{items}</ol>'

    def _html_appendix_references(self, data: Dict[str, Any]) -> str:
        """Render HTML appendix references."""
        refs = data.get("appendix_references", [])
        items = "".join(
            f'<li><strong>{ref.get("name", "-")}:</strong> {ref.get("description", "-")}</li>\n'
            for ref in refs
        )
        return f'<h2>10. Appendix References</h2>\n<ul>\n{items}</ul>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive overview."""
        o = data.get("executive_overview", {})
        return {
            "total_savings_mwh": o.get("total_savings_mwh", 0),
            "total_cost_savings": o.get("total_cost_savings", 0),
            "target_achievement_pct": o.get("target_achievement_pct", 0),
            "co2_avoided_tonnes": o.get("co2_avoided_tonnes", 0),
            "overall_status": o.get("overall_status", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        ecms = data.get("ecm_portfolio", [])
        risks = data.get("risk_assessment", [])
        return {
            "ecm_savings_pie": {
                "type": "pie",
                "labels": [e.get("name", "") for e in ecms],
                "values": [e.get("verified_mwh", 0) for e in ecms],
            },
            "ecm_realization_bar": {
                "type": "bar",
                "labels": [e.get("name", "") for e in ecms],
                "values": [e.get("realization_pct", 0) for e in ecms],
            },
            "risk_matrix": {
                "type": "scatter",
                "items": [
                    {
                        "risk": r.get("risk", ""),
                        "severity": r.get("severity", ""),
                        "likelihood": r.get("likelihood", ""),
                    }
                    for r in risks
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
            "h3{color:#495057;margin-top:20px;}"
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
