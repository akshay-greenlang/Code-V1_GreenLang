# -*- coding: utf-8 -*-
"""
AnnualMVReportTemplate - Annual M&V Summary Report for PACK-040.

Generates comprehensive annual M&V summary reports covering year-to-date
savings, cumulative savings across all reporting years, trend analysis
with performance indicators, compliance status against targets, and
multi-year performance comparison.

Sections:
    1. Annual Summary
    2. YTD Savings
    3. Monthly Savings Detail
    4. Cumulative Multi-Year Savings
    5. Trend Analysis
    6. ECM Performance
    7. Compliance Status
    8. Model Performance
    9. Key Findings
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (reporting requirements)
    - ISO 50015:2014 (M&V reporting)
    - FEMP M&V Guidelines 4.0
    - EU EED Article 7 (energy savings reporting)

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


class AnnualMVReportTemplate:
    """
    Annual M&V summary report template.

    Renders comprehensive annual M&V summary reports showing YTD savings,
    cumulative multi-year savings, trend analysis with performance
    indicators, compliance status, ECM-level performance tracking, and
    model validation status across markdown, HTML, and JSON formats.
    All outputs include SHA-256 provenance hashing for audit trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnnualMVReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render annual M&V report as Markdown.

        Args:
            data: Annual M&V reporting engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_annual_summary(data),
            self._md_ytd_savings(data),
            self._md_monthly_detail(data),
            self._md_cumulative_savings(data),
            self._md_trend_analysis(data),
            self._md_ecm_performance(data),
            self._md_compliance_status(data),
            self._md_model_performance(data),
            self._md_key_findings(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render annual M&V report as self-contained HTML.

        Args:
            data: Annual M&V reporting engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_annual_summary(data),
            self._html_ytd_savings(data),
            self._html_monthly_detail(data),
            self._html_cumulative_savings(data),
            self._html_trend_analysis(data),
            self._html_ecm_performance(data),
            self._html_compliance_status(data),
            self._html_model_performance(data),
            self._html_key_findings(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Annual M&amp;V Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render annual M&V report as structured JSON.

        Args:
            data: Annual M&V reporting engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "annual_mv_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "annual_summary": self._json_annual_summary(data),
            "ytd_savings": data.get("ytd_savings", {}),
            "monthly_detail": data.get("monthly_detail", []),
            "cumulative_savings": data.get("cumulative_savings", []),
            "trend_analysis": data.get("trend_analysis", {}),
            "ecm_performance": data.get("ecm_performance", []),
            "compliance_status": data.get("compliance_status", {}),
            "model_performance": data.get("model_performance", {}),
            "key_findings": data.get("key_findings", []),
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
        """Render markdown header with project metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Annual M&V Summary Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '-')}  \n"
            f"**Baseline Period:** {data.get('baseline_period', '-')}  \n"
            f"**IPMVP Option:** {data.get('ipmvp_option', '-')}  \n"
            f"**Year Number:** {data.get('year_number', 1)} of {data.get('contract_years', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 AnnualMVReportTemplate v40.0.0\n\n---"
        )

    def _md_annual_summary(self, data: Dict[str, Any]) -> str:
        """Render annual summary section."""
        s = data.get("annual_summary", {})
        return (
            "## 1. Annual Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Annual Savings | {self._format_energy(s.get('annual_savings_mwh', 0))} |\n"
            f"| Savings Target | {self._format_energy(s.get('target_savings_mwh', 0))} |\n"
            f"| Target Achievement | {self._fmt(s.get('target_achievement_pct', 0))}% |\n"
            f"| Cost Savings | {self._format_currency(s.get('annual_cost_savings', 0))} |\n"
            f"| Cumulative Savings | {self._format_energy(s.get('cumulative_savings_mwh', 0))} |\n"
            f"| CO2 Avoided | {self._fmt(s.get('co2_avoided_tonnes', 0), 1)} tCO2e |\n"
            f"| FSU | {self._fmt(s.get('fsu_pct', 0))}% |\n"
            f"| Model R-squared | {self._fmt(s.get('r_squared', 0), 4)} |\n"
            f"| Compliance | {s.get('compliance_status', '-')} |"
        )

    def _md_ytd_savings(self, data: Dict[str, Any]) -> str:
        """Render YTD savings section."""
        ytd = data.get("ytd_savings", {})
        if not ytd:
            return "## 2. Year-to-Date Savings\n\n_No YTD savings data available._"
        return (
            "## 2. Year-to-Date Savings\n\n"
            "| Component | Value |\n|-----------|-------|\n"
            f"| Adjusted Baseline | {self._format_energy(ytd.get('adjusted_baseline_mwh', 0))} |\n"
            f"| Actual Consumption | {self._format_energy(ytd.get('actual_consumption_mwh', 0))} |\n"
            f"| Avoided Energy | {self._format_energy(ytd.get('avoided_energy_mwh', 0))} |\n"
            f"| Routine Adjustments | {self._format_energy(ytd.get('routine_adj_mwh', 0))} |\n"
            f"| Non-Routine Adjustments | {self._format_energy(ytd.get('non_routine_adj_mwh', 0))} |\n"
            f"| Net Savings | {self._format_energy(ytd.get('net_savings_mwh', 0))} |\n"
            f"| Savings Rate | {self._fmt(ytd.get('savings_rate_pct', 0))}% |"
        )

    def _md_monthly_detail(self, data: Dict[str, Any]) -> str:
        """Render monthly savings detail section."""
        months = data.get("monthly_detail", [])
        if not months:
            return "## 3. Monthly Savings Detail\n\n_No monthly detail data available._"
        lines = [
            "## 3. Monthly Savings Detail\n",
            "| Month | Adj Baseline (MWh) | Actual (MWh) | Savings (MWh) | Savings (%) | Cumulative (MWh) |",
            "|-------|------------------:|------------:|-------------:|----------:|-----------------:|",
        ]
        for m in months:
            lines.append(
                f"| {m.get('month', '-')} "
                f"| {self._fmt(m.get('adjusted_baseline_mwh', 0), 1)} "
                f"| {self._fmt(m.get('actual_mwh', 0), 1)} "
                f"| {self._fmt(m.get('savings_mwh', 0), 1)} "
                f"| {self._fmt(m.get('savings_pct', 0))}% "
                f"| {self._fmt(m.get('cumulative_mwh', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_cumulative_savings(self, data: Dict[str, Any]) -> str:
        """Render cumulative multi-year savings section."""
        cumulative = data.get("cumulative_savings", [])
        if not cumulative:
            return "## 4. Cumulative Multi-Year Savings\n\n_No cumulative savings data available._"
        lines = [
            "## 4. Cumulative Multi-Year Savings\n",
            "| Year | Annual Savings (MWh) | Cumulative (MWh) | Annual Cost Savings | Cumulative Cost |",
            "|------|-------------------:|----------------:|-------------------:|----------------:|",
        ]
        for c in cumulative:
            lines.append(
                f"| {c.get('year', '-')} "
                f"| {self._fmt(c.get('annual_savings_mwh', 0), 1)} "
                f"| {self._fmt(c.get('cumulative_mwh', 0), 1)} "
                f"| {self._format_currency(c.get('annual_cost_savings', 0))} "
                f"| {self._format_currency(c.get('cumulative_cost_savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis section."""
        trend = data.get("trend_analysis", {})
        if not trend:
            return "## 5. Trend Analysis\n\n_No trend analysis data available._"
        return (
            "## 5. Trend Analysis\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Savings Trend | {trend.get('savings_trend', '-')} |\n"
            f"| YoY Change | {self._fmt(trend.get('yoy_change_pct', 0))}% |\n"
            f"| Moving Avg Savings | {self._format_energy(trend.get('moving_avg_savings_mwh', 0))} |\n"
            f"| Best Month | {trend.get('best_month', '-')} ({self._fmt(trend.get('best_month_savings_mwh', 0), 1)} MWh) |\n"
            f"| Worst Month | {trend.get('worst_month', '-')} ({self._fmt(trend.get('worst_month_savings_mwh', 0), 1)} MWh) |\n"
            f"| Seasonality | {trend.get('seasonality_pattern', '-')} |\n"
            f"| Performance Index | {self._fmt(trend.get('performance_index', 0), 3)} |"
        )

    def _md_ecm_performance(self, data: Dict[str, Any]) -> str:
        """Render ECM performance section."""
        ecms = data.get("ecm_performance", [])
        if not ecms:
            return "## 6. ECM Performance\n\n_No ECM performance data available._"
        lines = [
            "## 6. ECM Performance\n",
            "| ECM | Target (MWh) | Verified (MWh) | Realization (%) | Status |",
            "|-----|----------:|-------------:|---------------:|--------|",
        ]
        for ecm in ecms:
            status_indicator = ecm.get("status", "-")
            lines.append(
                f"| {ecm.get('ecm_name', '-')} "
                f"| {self._fmt(ecm.get('target_mwh', 0), 1)} "
                f"| {self._fmt(ecm.get('verified_mwh', 0), 1)} "
                f"| {self._fmt(ecm.get('realization_pct', 0))}% "
                f"| {status_indicator} |"
            )
        return "\n".join(lines)

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render compliance status section."""
        comp = data.get("compliance_status", {})
        if not comp:
            return "## 7. Compliance Status\n\n_No compliance status data available._"
        checks = comp.get("checks", [])
        lines = [
            "## 7. Compliance Status\n",
            f"**Overall Status:** {comp.get('overall_status', '-')}  \n"
            f"**Framework:** {comp.get('framework', '-')}  \n",
        ]
        if checks:
            lines.append("| Requirement | Status | Detail |")
            lines.append("|------------|:------:|--------|")
            for chk in checks:
                lines.append(
                    f"| {chk.get('requirement', '-')} "
                    f"| {chk.get('status', '-')} "
                    f"| {chk.get('detail', '-')} |"
                )
        return "\n".join(lines)

    def _md_model_performance(self, data: Dict[str, Any]) -> str:
        """Render model performance section."""
        model = data.get("model_performance", {})
        if not model:
            return "## 8. Model Performance\n\n_No model performance data available._"
        return (
            "## 8. Model Performance\n\n"
            "| Metric | Baseline | Current | Status |\n|--------|--------:|--------:|:------:|\n"
            f"| R-squared | {self._fmt(model.get('baseline_r_squared', 0), 4)} | {self._fmt(model.get('current_r_squared', 0), 4)} | {model.get('r_squared_status', '-')} |\n"
            f"| CVRMSE (%) | {self._fmt(model.get('baseline_cvrmse', 0), 1)} | {self._fmt(model.get('current_cvrmse', 0), 1)} | {model.get('cvrmse_status', '-')} |\n"
            f"| NMBE (%) | {self._fmt(model.get('baseline_nmbe', 0), 1)} | {self._fmt(model.get('current_nmbe', 0), 1)} | {model.get('nmbe_status', '-')} |\n"
            f"| Model Valid | - | - | {model.get('model_valid', '-')} |\n"
            f"| Refit Required | - | - | {model.get('refit_required', '-')} |"
        )

    def _md_key_findings(self, data: Dict[str, Any]) -> str:
        """Render key findings section."""
        findings = data.get("key_findings", [])
        if not findings:
            return "## 9. Key Findings\n\n_No key findings available._"
        lines = ["## 9. Key Findings\n"]
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. **{finding.get('title', 'Finding')}:** {finding.get('detail', '-')}")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Continue quarterly M&V reporting for ongoing savings verification",
                "Refit baseline model if CVRMSE exceeds ASHRAE 14 thresholds",
                "Investigate ECMs with realization below 80% for corrective action",
                "Update non-routine adjustment documentation annually",
            ]
        lines = ["## 10. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
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
            f'<h1>Annual M&amp;V Summary Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Year: {data.get("reporting_year", "-")} | '
            f'Option: {data.get("ipmvp_option", "-")}</p>'
        )

    def _html_annual_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML annual summary cards."""
        s = data.get("annual_summary", {})
        comp_cls = "severity-low" if s.get("target_achievement_pct", 0) >= 100 else "severity-medium"
        return (
            '<h2>1. Annual Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._fmt(s.get("annual_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Target</span>'
            f'<span class="value {comp_cls}">{self._fmt(s.get("target_achievement_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(s.get("annual_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Cumulative</span>'
            f'<span class="value">{self._fmt(s.get("cumulative_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Avoided</span>'
            f'<span class="value">{self._fmt(s.get("co2_avoided_tonnes", 0), 1)} t</span></div>\n'
            '</div>'
        )

    def _html_ytd_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML YTD savings table."""
        ytd = data.get("ytd_savings", {})
        return (
            '<h2>2. Year-to-Date Savings</h2>\n'
            '<table>\n'
            f'<tr><th>Component</th><th>Value</th></tr>\n'
            f'<tr><td>Adjusted Baseline</td><td>{self._format_energy(ytd.get("adjusted_baseline_mwh", 0))}</td></tr>\n'
            f'<tr><td>Actual Consumption</td><td>{self._format_energy(ytd.get("actual_consumption_mwh", 0))}</td></tr>\n'
            f'<tr><td>Avoided Energy</td><td>{self._format_energy(ytd.get("avoided_energy_mwh", 0))}</td></tr>\n'
            f'<tr><td>Net Savings</td><td><strong>{self._format_energy(ytd.get("net_savings_mwh", 0))}</strong></td></tr>\n'
            f'<tr><td>Savings Rate</td><td>{self._fmt(ytd.get("savings_rate_pct", 0))}%</td></tr>\n'
            '</table>'
        )

    def _html_monthly_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML monthly detail table."""
        months = data.get("monthly_detail", [])
        rows = ""
        for m in months:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("adjusted_baseline_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(m.get("actual_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(m.get("savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(m.get("savings_pct", 0))}%</td>'
                f'<td>{self._fmt(m.get("cumulative_mwh", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>3. Monthly Savings Detail</h2>\n'
            '<table>\n<tr><th>Month</th><th>Adj Baseline</th><th>Actual</th>'
            f'<th>Savings</th><th>Savings %</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_cumulative_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative multi-year savings table."""
        cumulative = data.get("cumulative_savings", [])
        rows = ""
        for c in cumulative:
            rows += (
                f'<tr><td>{c.get("year", "-")}</td>'
                f'<td>{self._fmt(c.get("annual_savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(c.get("cumulative_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(c.get("annual_cost_savings", 0))}</td>'
                f'<td>{self._format_currency(c.get("cumulative_cost_savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>4. Cumulative Multi-Year Savings</h2>\n'
            '<table>\n<tr><th>Year</th><th>Annual (MWh)</th><th>Cumulative (MWh)</th>'
            f'<th>Annual Cost</th><th>Cumulative Cost</th></tr>\n{rows}</table>'
        )

    def _html_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML trend analysis cards."""
        trend = data.get("trend_analysis", {})
        return (
            '<h2>5. Trend Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Trend</span>'
            f'<span class="value">{trend.get("savings_trend", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">YoY Change</span>'
            f'<span class="value">{self._fmt(trend.get("yoy_change_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Best Month</span>'
            f'<span class="value">{trend.get("best_month", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Perf Index</span>'
            f'<span class="value">{self._fmt(trend.get("performance_index", 0), 3)}</span></div>\n'
            '</div>'
        )

    def _html_ecm_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML ECM performance table."""
        ecms = data.get("ecm_performance", [])
        rows = ""
        for ecm in ecms:
            cls = "severity-low" if ecm.get("realization_pct", 0) >= 80 else "severity-high"
            rows += (
                f'<tr><td>{ecm.get("ecm_name", "-")}</td>'
                f'<td>{self._fmt(ecm.get("target_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(ecm.get("verified_mwh", 0), 1)}</td>'
                f'<td class="{cls}">{self._fmt(ecm.get("realization_pct", 0))}%</td>'
                f'<td>{ecm.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. ECM Performance</h2>\n'
            '<table>\n<tr><th>ECM</th><th>Target (MWh)</th><th>Verified (MWh)</th>'
            f'<th>Realization</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status table."""
        comp = data.get("compliance_status", {})
        checks = comp.get("checks", [])
        rows = ""
        for chk in checks:
            cls = "severity-low" if chk.get("status") == "PASS" else "severity-high"
            rows += (
                f'<tr><td>{chk.get("requirement", "-")}</td>'
                f'<td class="{cls}">{chk.get("status", "-")}</td>'
                f'<td>{chk.get("detail", "-")}</td></tr>\n'
            )
        return (
            '<h2>7. Compliance Status</h2>\n'
            f'<p>Overall: {comp.get("overall_status", "-")} | '
            f'Framework: {comp.get("framework", "-")}</p>\n'
            '<table>\n<tr><th>Requirement</th><th>Status</th>'
            f'<th>Detail</th></tr>\n{rows}</table>'
        )

    def _html_model_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML model performance cards."""
        model = data.get("model_performance", {})
        return (
            '<h2>8. Model Performance</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value">{self._fmt(model.get("current_r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">CVRMSE</span>'
            f'<span class="value">{self._fmt(model.get("current_cvrmse", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">NMBE</span>'
            f'<span class="value">{self._fmt(model.get("current_nmbe", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">Valid</span>'
            f'<span class="value">{model.get("model_valid", "-")}</span></div>\n'
            '</div>'
        )

    def _html_key_findings(self, data: Dict[str, Any]) -> str:
        """Render HTML key findings."""
        findings = data.get("key_findings", [])
        items = "".join(
            f'<li><strong>{f.get("title", "Finding")}:</strong> {f.get("detail", "-")}</li>\n'
            for f in findings
        )
        return f'<h2>9. Key Findings</h2>\n<ol>\n{items}</ol>'

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Continue quarterly M&V reporting for ongoing savings verification",
            "Refit baseline model if CVRMSE exceeds ASHRAE 14 thresholds",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_annual_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON annual summary."""
        s = data.get("annual_summary", {})
        return {
            "annual_savings_mwh": s.get("annual_savings_mwh", 0),
            "target_savings_mwh": s.get("target_savings_mwh", 0),
            "target_achievement_pct": s.get("target_achievement_pct", 0),
            "annual_cost_savings": s.get("annual_cost_savings", 0),
            "cumulative_savings_mwh": s.get("cumulative_savings_mwh", 0),
            "co2_avoided_tonnes": s.get("co2_avoided_tonnes", 0),
            "compliance_status": s.get("compliance_status", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        months = data.get("monthly_detail", [])
        cumulative = data.get("cumulative_savings", [])
        ecms = data.get("ecm_performance", [])
        return {
            "monthly_savings": {
                "type": "dual_bar",
                "labels": [m.get("month", "") for m in months],
                "series": {
                    "baseline": [m.get("adjusted_baseline_mwh", 0) for m in months],
                    "actual": [m.get("actual_mwh", 0) for m in months],
                },
            },
            "cumulative_trend": {
                "type": "line",
                "labels": [c.get("year", "") for c in cumulative],
                "values": [c.get("cumulative_mwh", 0) for c in cumulative],
            },
            "ecm_realization": {
                "type": "bar",
                "labels": [e.get("ecm_name", "") for e in ecms],
                "values": [e.get("realization_pct", 0) for e in ecms],
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
