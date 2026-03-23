# -*- coding: utf-8 -*-
"""
SavingsReportTemplate - Savings Verification Report for PACK-040.

Generates comprehensive savings verification reports covering adjusted
baseline versus actual consumption comparison, avoided energy calculation,
cost savings quantification, uncertainty bounds at specified confidence
levels, and cumulative savings tracking over reporting periods.

Sections:
    1. Savings Summary
    2. Adjusted Baseline vs Actual
    3. Avoided Energy
    4. Cost Savings
    5. Routine Adjustments
    6. Non-Routine Adjustments
    7. Uncertainty Analysis
    8. Cumulative Savings
    9. Savings Breakdown by ECM
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (savings determination)
    - ASHRAE Guideline 14-2014 (savings uncertainty)
    - ISO 50015:2014 (M&V of energy performance)
    - FEMP M&V Guidelines 4.0

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


class SavingsReportTemplate:
    """
    Savings verification report template.

    Renders comprehensive savings verification reports showing adjusted
    baseline versus actual consumption, avoided energy with routine
    and non-routine adjustments, cost savings quantification, uncertainty
    bounds at specified confidence levels, and cumulative savings
    tracking across markdown, HTML, and JSON formats. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SavingsReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render savings verification report as Markdown.

        Args:
            data: Savings engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_savings_summary(data),
            self._md_adjusted_vs_actual(data),
            self._md_avoided_energy(data),
            self._md_cost_savings(data),
            self._md_routine_adjustments(data),
            self._md_non_routine_adjustments(data),
            self._md_uncertainty_analysis(data),
            self._md_cumulative_savings(data),
            self._md_savings_by_ecm(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render savings verification report as self-contained HTML.

        Args:
            data: Savings engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_savings_summary(data),
            self._html_adjusted_vs_actual(data),
            self._html_avoided_energy(data),
            self._html_cost_savings(data),
            self._html_routine_adjustments(data),
            self._html_non_routine_adjustments(data),
            self._html_uncertainty_analysis(data),
            self._html_cumulative_savings(data),
            self._html_savings_by_ecm(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Savings Verification Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render savings verification report as structured JSON.

        Args:
            data: Savings engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "savings_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "savings_summary": self._json_savings_summary(data),
            "adjusted_vs_actual": data.get("adjusted_vs_actual", []),
            "avoided_energy": data.get("avoided_energy", {}),
            "cost_savings": data.get("cost_savings", {}),
            "routine_adjustments": data.get("routine_adjustments", []),
            "non_routine_adjustments": data.get("non_routine_adjustments", []),
            "uncertainty_analysis": data.get("uncertainty_analysis", {}),
            "cumulative_savings": data.get("cumulative_savings", []),
            "savings_by_ecm": data.get("savings_by_ecm", []),
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
            f"# Savings Verification Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Baseline Period:** {data.get('baseline_period', '-')}  \n"
            f"**IPMVP Option:** {data.get('ipmvp_option', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 SavingsReportTemplate v40.0.0\n\n---"
        )

    def _md_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render savings summary section."""
        s = data.get("savings_summary", {})
        return (
            "## 1. Savings Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Adjusted Baseline | {self._format_energy(s.get('adjusted_baseline_mwh', 0))} |\n"
            f"| Actual Consumption | {self._format_energy(s.get('actual_consumption_mwh', 0))} |\n"
            f"| Avoided Energy | {self._format_energy(s.get('avoided_energy_mwh', 0))} |\n"
            f"| Savings Percentage | {self._fmt(s.get('savings_pct', 0))}% |\n"
            f"| Cost Savings | {self._format_currency(s.get('cost_savings', 0))} |\n"
            f"| Uncertainty (+/-) | {self._format_energy(s.get('uncertainty_mwh', 0))} |\n"
            f"| Confidence Level | {self._fmt(s.get('confidence_level_pct', 90))}% |\n"
            f"| Savings Significant | {s.get('is_significant', '-')} |"
        )

    def _md_adjusted_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render adjusted baseline vs actual section."""
        periods = data.get("adjusted_vs_actual", [])
        if not periods:
            return "## 2. Adjusted Baseline vs Actual\n\n_No period data available._"
        lines = [
            "## 2. Adjusted Baseline vs Actual\n",
            "| Period | Adjusted Baseline (MWh) | Actual (MWh) | Savings (MWh) | Savings (%) |",
            "|--------|---------------------:|------------:|-------------:|----------:|",
        ]
        for p in periods:
            lines.append(
                f"| {p.get('period', '-')} "
                f"| {self._fmt(p.get('adjusted_baseline_mwh', 0), 1)} "
                f"| {self._fmt(p.get('actual_mwh', 0), 1)} "
                f"| {self._fmt(p.get('savings_mwh', 0), 1)} "
                f"| {self._fmt(p.get('savings_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_avoided_energy(self, data: Dict[str, Any]) -> str:
        """Render avoided energy section."""
        avoided = data.get("avoided_energy", {})
        if not avoided:
            return "## 3. Avoided Energy\n\n_No avoided energy data available._"
        return (
            "## 3. Avoided Energy\n\n"
            "| Component | Value |\n|-----------|-------|\n"
            f"| Baseline Consumption | {self._format_energy(avoided.get('baseline_mwh', 0))} |\n"
            f"| Routine Adjustments | {self._format_energy(avoided.get('routine_adj_mwh', 0))} |\n"
            f"| Non-Routine Adjustments | {self._format_energy(avoided.get('non_routine_adj_mwh', 0))} |\n"
            f"| Adjusted Baseline | {self._format_energy(avoided.get('adjusted_baseline_mwh', 0))} |\n"
            f"| Reporting Period Consumption | {self._format_energy(avoided.get('reporting_consumption_mwh', 0))} |\n"
            f"| **Avoided Energy** | **{self._format_energy(avoided.get('avoided_energy_mwh', 0))}** |\n"
            f"| Normalized Savings | {self._format_energy(avoided.get('normalized_savings_mwh', 0))} |\n"
            f"| CO2 Avoided | {self._fmt(avoided.get('co2_avoided_tonnes', 0), 1)} tCO2e |"
        )

    def _md_cost_savings(self, data: Dict[str, Any]) -> str:
        """Render cost savings section."""
        cost = data.get("cost_savings", {})
        if not cost:
            return "## 4. Cost Savings\n\n_No cost savings data available._"
        breakdown = cost.get("breakdown", [])
        lines = [
            "## 4. Cost Savings\n",
            f"**Total Cost Savings:** {self._format_currency(cost.get('total_cost_savings', 0))}  \n"
            f"**Energy Rate:** {self._format_currency(cost.get('energy_rate_per_mwh', 0))}/MWh  \n"
            f"**Demand Savings:** {self._format_currency(cost.get('demand_savings', 0))}  \n"
            f"**Simple Payback:** {self._fmt(cost.get('simple_payback_years', 0), 1)} years  \n"
            f"**ROI:** {self._fmt(cost.get('roi_pct', 0))}%\n",
        ]
        if breakdown:
            lines.append("### Cost Breakdown by ECM\n")
            lines.append("| ECM | Energy Savings | Cost Savings | Share (%) |")
            lines.append("|-----|-------------:|-------------:|--------:|")
            for b in breakdown:
                lines.append(
                    f"| {b.get('ecm_name', '-')} "
                    f"| {self._format_energy(b.get('energy_savings_mwh', 0))} "
                    f"| {self._format_currency(b.get('cost_savings', 0))} "
                    f"| {self._fmt(b.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_routine_adjustments(self, data: Dict[str, Any]) -> str:
        """Render routine adjustments section."""
        adjustments = data.get("routine_adjustments", [])
        if not adjustments:
            return "## 5. Routine Adjustments\n\n_No routine adjustment data available._"
        lines = [
            "## 5. Routine Adjustments\n",
            "| Variable | Baseline | Reporting | Adjustment (MWh) | Method |",
            "|----------|--------:|---------:|---------------:|--------|",
        ]
        for adj in adjustments:
            lines.append(
                f"| {adj.get('variable', '-')} "
                f"| {self._fmt(adj.get('baseline_value', 0), 2)} "
                f"| {self._fmt(adj.get('reporting_value', 0), 2)} "
                f"| {self._fmt(adj.get('adjustment_mwh', 0), 1)} "
                f"| {adj.get('method', '-')} |"
            )
        total_adj = sum(a.get("adjustment_mwh", 0) for a in adjustments)
        lines.append(f"| **Total** | - | - | **{self._fmt(total_adj, 1)}** | - |")
        return "\n".join(lines)

    def _md_non_routine_adjustments(self, data: Dict[str, Any]) -> str:
        """Render non-routine adjustments section."""
        adjustments = data.get("non_routine_adjustments", [])
        if not adjustments:
            return "## 6. Non-Routine Adjustments\n\n_No non-routine adjustment data available._"
        lines = [
            "## 6. Non-Routine Adjustments\n",
            "| Event | Date | Impact (MWh) | Method | Verified |",
            "|-------|------|----------:|--------|:--------:|",
        ]
        for adj in adjustments:
            lines.append(
                f"| {adj.get('event', '-')} "
                f"| {adj.get('date', '-')} "
                f"| {self._fmt(adj.get('impact_mwh', 0), 1)} "
                f"| {adj.get('method', '-')} "
                f"| {adj.get('verified', '-')} |"
            )
        return "\n".join(lines)

    def _md_uncertainty_analysis(self, data: Dict[str, Any]) -> str:
        """Render uncertainty analysis section."""
        unc = data.get("uncertainty_analysis", {})
        if not unc:
            return "## 7. Uncertainty Analysis\n\n_No uncertainty analysis data available._"
        return (
            "## 7. Uncertainty Analysis\n\n"
            "| Component | Value |\n|-----------|-------|\n"
            f"| Model Uncertainty | {self._fmt(unc.get('model_uncertainty_pct', 0))}% |\n"
            f"| Measurement Uncertainty | {self._fmt(unc.get('measurement_uncertainty_pct', 0))}% |\n"
            f"| Sampling Uncertainty | {self._fmt(unc.get('sampling_uncertainty_pct', 0))}% |\n"
            f"| Combined Uncertainty | {self._fmt(unc.get('combined_uncertainty_pct', 0))}% |\n"
            f"| FSU at {self._fmt(unc.get('confidence_level_pct', 90))}% | {self._fmt(unc.get('fsu_pct', 0))}% |\n"
            f"| Savings +/- | {self._format_energy(unc.get('uncertainty_mwh', 0))} |\n"
            f"| Lower Bound | {self._format_energy(unc.get('lower_bound_mwh', 0))} |\n"
            f"| Upper Bound | {self._format_energy(unc.get('upper_bound_mwh', 0))} |\n"
            f"| Significant | {unc.get('is_significant', '-')} |"
        )

    def _md_cumulative_savings(self, data: Dict[str, Any]) -> str:
        """Render cumulative savings section."""
        cumulative = data.get("cumulative_savings", [])
        if not cumulative:
            return "## 8. Cumulative Savings\n\n_No cumulative savings data available._"
        lines = [
            "## 8. Cumulative Savings\n",
            "| Period | Period Savings (MWh) | Cumulative (MWh) | Cumulative Cost |",
            "|--------|------------------:|----------------:|---------------:|",
        ]
        for c in cumulative:
            lines.append(
                f"| {c.get('period', '-')} "
                f"| {self._fmt(c.get('period_savings_mwh', 0), 1)} "
                f"| {self._fmt(c.get('cumulative_mwh', 0), 1)} "
                f"| {self._format_currency(c.get('cumulative_cost_savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_savings_by_ecm(self, data: Dict[str, Any]) -> str:
        """Render savings by ECM section."""
        ecm_savings = data.get("savings_by_ecm", [])
        if not ecm_savings:
            return "## 9. Savings Breakdown by ECM\n\n_No ECM savings data available._"
        lines = [
            "## 9. Savings Breakdown by ECM\n",
            "| ECM | Estimated (MWh) | Verified (MWh) | Realization (%) | Status |",
            "|-----|-------------:|-------------:|---------------:|--------|",
        ]
        for ecm in ecm_savings:
            lines.append(
                f"| {ecm.get('ecm_name', '-')} "
                f"| {self._fmt(ecm.get('estimated_mwh', 0), 1)} "
                f"| {self._fmt(ecm.get('verified_mwh', 0), 1)} "
                f"| {self._fmt(ecm.get('realization_pct', 0))}% "
                f"| {ecm.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Monitor savings persistence quarterly to detect degradation",
                "Investigate ECMs with realization rate below 80%",
                "Update baseline model if non-routine events exceed 10% of baseline",
                "Verify metering accuracy before next reporting period",
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
            f'<h1>Savings Verification Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Reporting: {data.get("reporting_period", "-")} | '
            f'Option: {data.get("ipmvp_option", "-")}</p>'
        )

    def _html_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML savings summary cards."""
        s = data.get("savings_summary", {})
        sig_cls = "severity-low" if s.get("is_significant") else "severity-high"
        return (
            '<h2>1. Savings Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Avoided Energy</span>'
            f'<span class="value">{self._fmt(s.get("avoided_energy_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Savings %</span>'
            f'<span class="value">{self._fmt(s.get("savings_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(s.get("cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Uncertainty</span>'
            f'<span class="value">+/- {self._fmt(s.get("uncertainty_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Significant</span>'
            f'<span class="value {sig_cls}">{s.get("is_significant", "-")}</span></div>\n'
            '</div>'
        )

    def _html_adjusted_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render HTML adjusted vs actual table."""
        periods = data.get("adjusted_vs_actual", [])
        rows = ""
        for p in periods:
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{self._fmt(p.get("adjusted_baseline_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("actual_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("savings_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>2. Adjusted Baseline vs Actual</h2>\n'
            '<table>\n<tr><th>Period</th><th>Adj Baseline (MWh)</th><th>Actual (MWh)</th>'
            f'<th>Savings (MWh)</th><th>Savings (%)</th></tr>\n{rows}</table>'
        )

    def _html_avoided_energy(self, data: Dict[str, Any]) -> str:
        """Render HTML avoided energy breakdown."""
        avoided = data.get("avoided_energy", {})
        return (
            '<h2>3. Avoided Energy</h2>\n'
            '<table>\n'
            f'<tr><th>Component</th><th>Value</th></tr>\n'
            f'<tr><td>Baseline Consumption</td><td>{self._format_energy(avoided.get("baseline_mwh", 0))}</td></tr>\n'
            f'<tr><td>Routine Adjustments</td><td>{self._format_energy(avoided.get("routine_adj_mwh", 0))}</td></tr>\n'
            f'<tr><td>Non-Routine Adjustments</td><td>{self._format_energy(avoided.get("non_routine_adj_mwh", 0))}</td></tr>\n'
            f'<tr><td>Adjusted Baseline</td><td>{self._format_energy(avoided.get("adjusted_baseline_mwh", 0))}</td></tr>\n'
            f'<tr><td>Reporting Consumption</td><td>{self._format_energy(avoided.get("reporting_consumption_mwh", 0))}</td></tr>\n'
            f'<tr><td><strong>Avoided Energy</strong></td>'
            f'<td><strong>{self._format_energy(avoided.get("avoided_energy_mwh", 0))}</strong></td></tr>\n'
            '</table>'
        )

    def _html_cost_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML cost savings cards."""
        cost = data.get("cost_savings", {})
        return (
            '<h2>4. Cost Savings</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Savings</span>'
            f'<span class="value">{self._format_currency(cost.get("total_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Energy Rate</span>'
            f'<span class="value">{self._format_currency(cost.get("energy_rate_per_mwh", 0))}/MWh</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(cost.get("simple_payback_years", 0), 1)} yrs</span></div>\n'
            f'  <div class="card"><span class="label">ROI</span>'
            f'<span class="value">{self._fmt(cost.get("roi_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_routine_adjustments(self, data: Dict[str, Any]) -> str:
        """Render HTML routine adjustments table."""
        adjustments = data.get("routine_adjustments", [])
        rows = ""
        for adj in adjustments:
            rows += (
                f'<tr><td>{adj.get("variable", "-")}</td>'
                f'<td>{self._fmt(adj.get("baseline_value", 0), 2)}</td>'
                f'<td>{self._fmt(adj.get("reporting_value", 0), 2)}</td>'
                f'<td>{self._fmt(adj.get("adjustment_mwh", 0), 1)}</td>'
                f'<td>{adj.get("method", "-")}</td></tr>\n'
            )
        return (
            '<h2>5. Routine Adjustments</h2>\n'
            '<table>\n<tr><th>Variable</th><th>Baseline</th><th>Reporting</th>'
            f'<th>Adjustment (MWh)</th><th>Method</th></tr>\n{rows}</table>'
        )

    def _html_non_routine_adjustments(self, data: Dict[str, Any]) -> str:
        """Render HTML non-routine adjustments table."""
        adjustments = data.get("non_routine_adjustments", [])
        rows = ""
        for adj in adjustments:
            rows += (
                f'<tr><td>{adj.get("event", "-")}</td>'
                f'<td>{adj.get("date", "-")}</td>'
                f'<td>{self._fmt(adj.get("impact_mwh", 0), 1)}</td>'
                f'<td>{adj.get("method", "-")}</td>'
                f'<td>{adj.get("verified", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Non-Routine Adjustments</h2>\n'
            '<table>\n<tr><th>Event</th><th>Date</th><th>Impact (MWh)</th>'
            f'<th>Method</th><th>Verified</th></tr>\n{rows}</table>'
        )

    def _html_uncertainty_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty analysis cards."""
        unc = data.get("uncertainty_analysis", {})
        return (
            '<h2>7. Uncertainty Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Model Unc.</span>'
            f'<span class="value">{self._fmt(unc.get("model_uncertainty_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Measurement Unc.</span>'
            f'<span class="value">{self._fmt(unc.get("measurement_uncertainty_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Combined</span>'
            f'<span class="value">{self._fmt(unc.get("combined_uncertainty_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">FSU</span>'
            f'<span class="value">{self._fmt(unc.get("fsu_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_cumulative_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative savings table."""
        cumulative = data.get("cumulative_savings", [])
        rows = ""
        for c in cumulative:
            rows += (
                f'<tr><td>{c.get("period", "-")}</td>'
                f'<td>{self._fmt(c.get("period_savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(c.get("cumulative_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(c.get("cumulative_cost_savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>8. Cumulative Savings</h2>\n'
            '<table>\n<tr><th>Period</th><th>Period Savings (MWh)</th>'
            f'<th>Cumulative (MWh)</th><th>Cumulative Cost</th></tr>\n{rows}</table>'
        )

    def _html_savings_by_ecm(self, data: Dict[str, Any]) -> str:
        """Render HTML savings by ECM table."""
        ecm_savings = data.get("savings_by_ecm", [])
        rows = ""
        for ecm in ecm_savings:
            cls = "severity-low" if ecm.get("realization_pct", 0) >= 80 else "severity-high"
            rows += (
                f'<tr><td>{ecm.get("ecm_name", "-")}</td>'
                f'<td>{self._fmt(ecm.get("estimated_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(ecm.get("verified_mwh", 0), 1)}</td>'
                f'<td class="{cls}">{self._fmt(ecm.get("realization_pct", 0))}%</td>'
                f'<td>{ecm.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Savings by ECM</h2>\n'
            '<table>\n<tr><th>ECM</th><th>Estimated (MWh)</th><th>Verified (MWh)</th>'
            f'<th>Realization (%)</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Monitor savings persistence quarterly to detect degradation",
            "Investigate ECMs with realization rate below 80%",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_savings_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON savings summary."""
        s = data.get("savings_summary", {})
        return {
            "adjusted_baseline_mwh": s.get("adjusted_baseline_mwh", 0),
            "actual_consumption_mwh": s.get("actual_consumption_mwh", 0),
            "avoided_energy_mwh": s.get("avoided_energy_mwh", 0),
            "savings_pct": s.get("savings_pct", 0),
            "cost_savings": s.get("cost_savings", 0),
            "uncertainty_mwh": s.get("uncertainty_mwh", 0),
            "confidence_level_pct": s.get("confidence_level_pct", 90),
            "is_significant": s.get("is_significant", False),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        periods = data.get("adjusted_vs_actual", [])
        cumulative = data.get("cumulative_savings", [])
        ecm_savings = data.get("savings_by_ecm", [])
        return {
            "baseline_vs_actual": {
                "type": "dual_line",
                "labels": [p.get("period", "") for p in periods],
                "series": {
                    "adjusted_baseline": [p.get("adjusted_baseline_mwh", 0) for p in periods],
                    "actual": [p.get("actual_mwh", 0) for p in periods],
                },
            },
            "cumulative_savings": {
                "type": "area",
                "labels": [c.get("period", "") for c in cumulative],
                "values": [c.get("cumulative_mwh", 0) for c in cumulative],
            },
            "ecm_realization": {
                "type": "bar",
                "labels": [e.get("ecm_name", "") for e in ecm_savings],
                "values": [e.get("realization_pct", 0) for e in ecm_savings],
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
