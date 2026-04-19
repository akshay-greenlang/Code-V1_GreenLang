# -*- coding: utf-8 -*-
"""
PersistenceReportTemplate - Persistence Tracking Report for PACK-040.

Generates comprehensive savings persistence tracking reports covering
year-over-year savings trends, degradation analysis with rate
calculation, persistence factors, equipment performance decay, and
re-commissioning recommendations for sustained energy savings.

Sections:
    1. Persistence Summary
    2. Year-over-Year Savings
    3. Degradation Analysis
    4. Persistence Factors
    5. Equipment Performance
    6. Operational Changes
    7. Model Stability
    8. Re-commissioning Assessment
    9. Risk Indicators
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (persistence M&V)
    - ASHRAE Guideline 14-2014 (ongoing savings)
    - ISO 50015:2014 (long-term M&V)
    - FEMP M&V Guidelines 4.0 (persistence)

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


class PersistenceReportTemplate:
    """
    Persistence tracking report template.

    Renders comprehensive savings persistence tracking reports showing
    YoY savings trends, degradation analysis with rate calculation,
    persistence factors by ECM, equipment performance decay curves,
    operational changes impacting savings, model stability assessment,
    and re-commissioning recommendations across markdown, HTML, and
    JSON formats. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PersistenceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render persistence tracking report as Markdown.

        Args:
            data: Persistence engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_persistence_summary(data),
            self._md_yoy_savings(data),
            self._md_degradation_analysis(data),
            self._md_persistence_factors(data),
            self._md_equipment_performance(data),
            self._md_operational_changes(data),
            self._md_model_stability(data),
            self._md_recommissioning(data),
            self._md_risk_indicators(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render persistence tracking report as self-contained HTML.

        Args:
            data: Persistence engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_persistence_summary(data),
            self._html_yoy_savings(data),
            self._html_degradation_analysis(data),
            self._html_persistence_factors(data),
            self._html_equipment_performance(data),
            self._html_operational_changes(data),
            self._html_model_stability(data),
            self._html_recommissioning(data),
            self._html_risk_indicators(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Persistence Tracking Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render persistence tracking report as structured JSON.

        Args:
            data: Persistence engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "persistence_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "persistence_summary": self._json_persistence_summary(data),
            "yoy_savings": data.get("yoy_savings", []),
            "degradation_analysis": data.get("degradation_analysis", {}),
            "persistence_factors": data.get("persistence_factors", []),
            "equipment_performance": data.get("equipment_performance", []),
            "operational_changes": data.get("operational_changes", []),
            "model_stability": data.get("model_stability", {}),
            "recommissioning": data.get("recommissioning", {}),
            "risk_indicators": data.get("risk_indicators", []),
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
            f"# Persistence Tracking Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Tracking Period:** {data.get('tracking_period', '-')}  \n"
            f"**Years Since Implementation:** {data.get('years_since_impl', 0)}  \n"
            f"**Total ECMs:** {data.get('total_ecms', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 PersistenceReportTemplate v40.0.0\n\n---"
        )

    def _md_persistence_summary(self, data: Dict[str, Any]) -> str:
        """Render persistence summary section."""
        s = data.get("persistence_summary", {})
        return (
            "## 1. Persistence Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Year 1 Savings | {self._format_energy(s.get('year1_savings_mwh', 0))} |\n"
            f"| Current Year Savings | {self._format_energy(s.get('current_savings_mwh', 0))} |\n"
            f"| Persistence Factor | {self._fmt(s.get('persistence_factor', 0), 3)} |\n"
            f"| Degradation Rate | {self._fmt(s.get('degradation_rate_pct', 0))}% per year |\n"
            f"| Cumulative Savings | {self._format_energy(s.get('cumulative_savings_mwh', 0))} |\n"
            f"| Cumulative Cost Savings | {self._format_currency(s.get('cumulative_cost_savings', 0))} |\n"
            f"| ECMs At Risk | {s.get('ecms_at_risk', 0)} of {s.get('total_ecms', 0)} |\n"
            f"| Re-commissioning Needed | {s.get('recommissioning_needed', '-')} |\n"
            f"| Overall Status | {s.get('overall_status', '-')} |"
        )

    def _md_yoy_savings(self, data: Dict[str, Any]) -> str:
        """Render year-over-year savings section."""
        yoy = data.get("yoy_savings", [])
        if not yoy:
            return "## 2. Year-over-Year Savings\n\n_No YoY savings data available._"
        lines = [
            "## 2. Year-over-Year Savings\n",
            "| Year | Savings (MWh) | Change (%) | Persistence Factor | Cumulative (MWh) |",
            "|------|------------:|--------:|------------------:|-----------------:|",
        ]
        for y in yoy:
            lines.append(
                f"| {y.get('year', '-')} "
                f"| {self._fmt(y.get('savings_mwh', 0), 1)} "
                f"| {self._fmt(y.get('change_pct', 0))}% "
                f"| {self._fmt(y.get('persistence_factor', 0), 3)} "
                f"| {self._fmt(y.get('cumulative_mwh', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_degradation_analysis(self, data: Dict[str, Any]) -> str:
        """Render degradation analysis section."""
        deg = data.get("degradation_analysis", {})
        if not deg:
            return "## 3. Degradation Analysis\n\n_No degradation analysis data available._"
        return (
            "## 3. Degradation Analysis\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Degradation Model | {deg.get('model', '-')} |\n"
            f"| Annual Degradation | {self._fmt(deg.get('annual_degradation_pct', 0))}% |\n"
            f"| Projected Year 5 | {self._format_energy(deg.get('projected_year5_mwh', 0))} |\n"
            f"| Projected Year 10 | {self._format_energy(deg.get('projected_year10_mwh', 0))} |\n"
            f"| Half-Life (years) | {self._fmt(deg.get('half_life_years', 0), 1)} |\n"
            f"| R-squared of Fit | {self._fmt(deg.get('r_squared', 0), 4)} |\n"
            f"| Primary Cause | {deg.get('primary_cause', '-')} |\n"
            f"| Secondary Cause | {deg.get('secondary_cause', '-')} |"
        )

    def _md_persistence_factors(self, data: Dict[str, Any]) -> str:
        """Render persistence factors by ECM section."""
        factors = data.get("persistence_factors", [])
        if not factors:
            return "## 4. Persistence Factors\n\n_No persistence factor data available._"
        lines = [
            "## 4. Persistence Factors by ECM\n",
            "| ECM | Year 1 (MWh) | Current (MWh) | Factor | Degradation (%) | Status |",
            "|-----|----------:|-------------:|------:|--------------:|--------|",
        ]
        for f in factors:
            status = f.get("status", "-")
            lines.append(
                f"| {f.get('ecm_name', '-')} "
                f"| {self._fmt(f.get('year1_mwh', 0), 1)} "
                f"| {self._fmt(f.get('current_mwh', 0), 1)} "
                f"| {self._fmt(f.get('factor', 0), 3)} "
                f"| {self._fmt(f.get('degradation_pct', 0))}% "
                f"| {status} |"
            )
        return "\n".join(lines)

    def _md_equipment_performance(self, data: Dict[str, Any]) -> str:
        """Render equipment performance section."""
        equipment = data.get("equipment_performance", [])
        if not equipment:
            return "## 5. Equipment Performance\n\n_No equipment performance data available._"
        lines = [
            "## 5. Equipment Performance\n",
            "| Equipment | Design Efficiency | Current Efficiency | Decay (%) | Maintenance |",
            "|-----------|-----------------:|------------------:|--------:|-------------|",
        ]
        for eq in equipment:
            lines.append(
                f"| {eq.get('name', '-')} "
                f"| {self._fmt(eq.get('design_efficiency', 0))}% "
                f"| {self._fmt(eq.get('current_efficiency', 0))}% "
                f"| {self._fmt(eq.get('decay_pct', 0))}% "
                f"| {eq.get('maintenance_status', '-')} |"
            )
        return "\n".join(lines)

    def _md_operational_changes(self, data: Dict[str, Any]) -> str:
        """Render operational changes section."""
        changes = data.get("operational_changes", [])
        if not changes:
            return "## 6. Operational Changes\n\n_No operational changes recorded._"
        lines = [
            "## 6. Operational Changes\n",
            "| Date | Change | Impact (MWh) | Adjustment Required | Status |",
            "|------|--------|----------:|:-------------------:|--------|",
        ]
        for c in changes:
            lines.append(
                f"| {c.get('date', '-')} "
                f"| {c.get('description', '-')} "
                f"| {self._fmt(c.get('impact_mwh', 0), 1)} "
                f"| {c.get('adjustment_required', '-')} "
                f"| {c.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_model_stability(self, data: Dict[str, Any]) -> str:
        """Render model stability section."""
        model = data.get("model_stability", {})
        if not model:
            return "## 7. Model Stability\n\n_No model stability data available._"
        return (
            "## 7. Model Stability\n\n"
            "| Metric | Baseline | Current | Change | Status |\n|--------|--------:|--------:|-------:|:------:|\n"
            f"| R-squared | {self._fmt(model.get('baseline_r_squared', 0), 4)} | {self._fmt(model.get('current_r_squared', 0), 4)} | {self._fmt(model.get('r_squared_change', 0), 4)} | {model.get('r_squared_status', '-')} |\n"
            f"| CVRMSE (%) | {self._fmt(model.get('baseline_cvrmse', 0), 1)} | {self._fmt(model.get('current_cvrmse', 0), 1)} | {self._fmt(model.get('cvrmse_change', 0), 1)} | {model.get('cvrmse_status', '-')} |\n"
            f"| Coefficients | - | - | {model.get('coefficient_stability', '-')} | {model.get('coefficient_status', '-')} |\n"
            f"| Residual Pattern | - | - | {model.get('residual_pattern', '-')} | {model.get('residual_status', '-')} |\n"
            f"| Refit Recommended | - | - | - | {model.get('refit_recommended', '-')} |"
        )

    def _md_recommissioning(self, data: Dict[str, Any]) -> str:
        """Render re-commissioning assessment section."""
        recom = data.get("recommissioning", {})
        if not recom:
            return "## 8. Re-commissioning Assessment\n\n_No re-commissioning data available._"
        actions = recom.get("actions", [])
        lines = [
            "## 8. Re-commissioning Assessment\n",
            f"**Assessment Result:** {recom.get('assessment_result', '-')}  \n"
            f"**Priority:** {recom.get('priority', '-')}  \n"
            f"**Estimated Recovery:** {self._format_energy(recom.get('estimated_recovery_mwh', 0))}  \n"
            f"**Estimated Cost:** {self._format_currency(recom.get('estimated_cost', 0))}  \n"
            f"**Payback:** {self._fmt(recom.get('payback_months', 0), 0)} months  \n",
        ]
        if actions:
            lines.append("### Recommended Actions\n")
            lines.append("| Action | ECM | Priority | Recovery (MWh) | Cost |")
            lines.append("|--------|-----|----------|-------------:|-----:|")
            for a in actions:
                lines.append(
                    f"| {a.get('action', '-')} "
                    f"| {a.get('ecm', '-')} "
                    f"| {a.get('priority', '-')} "
                    f"| {self._fmt(a.get('recovery_mwh', 0), 1)} "
                    f"| {self._format_currency(a.get('cost', 0))} |"
                )
        return "\n".join(lines)

    def _md_risk_indicators(self, data: Dict[str, Any]) -> str:
        """Render risk indicators section."""
        risks = data.get("risk_indicators", [])
        if not risks:
            return "## 9. Risk Indicators\n\n_No risk indicator data available._"
        lines = [
            "## 9. Risk Indicators\n",
            "| Indicator | Value | Threshold | Status |",
            "|-----------|------:|----------:|:------:|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('indicator', '-')} "
                f"| {self._fmt(r.get('value', 0), 2)} "
                f"| {self._fmt(r.get('threshold', 0), 2)} "
                f"| {r.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Schedule re-commissioning for ECMs with persistence factor below 0.80",
                "Implement preventive maintenance to reduce equipment degradation",
                "Update baseline model when operational changes exceed 10% impact",
                "Conduct annual persistence review for all performance contracts",
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
            f'<h1>Persistence Tracking Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("tracking_period", "-")} | '
            f'Years: {data.get("years_since_impl", 0)}</p>'
        )

    def _html_persistence_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML persistence summary cards."""
        s = data.get("persistence_summary", {})
        factor_cls = "severity-low" if s.get("persistence_factor", 0) >= 0.8 else "severity-high"
        return (
            '<h2>1. Persistence Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Current Savings</span>'
            f'<span class="value">{self._fmt(s.get("current_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Persistence</span>'
            f'<span class="value {factor_cls}">{self._fmt(s.get("persistence_factor", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Degradation</span>'
            f'<span class="value">{self._fmt(s.get("degradation_rate_pct", 0))}%/yr</span></div>\n'
            f'  <div class="card"><span class="label">Cumulative</span>'
            f'<span class="value">{self._fmt(s.get("cumulative_savings_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">At Risk</span>'
            f'<span class="value">{s.get("ecms_at_risk", 0)}/{s.get("total_ecms", 0)}</span></div>\n'
            '</div>'
        )

    def _html_yoy_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML YoY savings table."""
        yoy = data.get("yoy_savings", [])
        rows = ""
        for y in yoy:
            rows += (
                f'<tr><td>{y.get("year", "-")}</td>'
                f'<td>{self._fmt(y.get("savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(y.get("change_pct", 0))}%</td>'
                f'<td>{self._fmt(y.get("persistence_factor", 0), 3)}</td>'
                f'<td>{self._fmt(y.get("cumulative_mwh", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>2. Year-over-Year Savings</h2>\n'
            '<table>\n<tr><th>Year</th><th>Savings (MWh)</th><th>Change</th>'
            f'<th>Persistence</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_degradation_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML degradation analysis cards."""
        deg = data.get("degradation_analysis", {})
        return (
            '<h2>3. Degradation Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Rate</span>'
            f'<span class="value">{self._fmt(deg.get("annual_degradation_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Half-Life</span>'
            f'<span class="value">{self._fmt(deg.get("half_life_years", 0), 1)} yrs</span></div>\n'
            f'  <div class="card"><span class="label">Year 5 Proj.</span>'
            f'<span class="value">{self._fmt(deg.get("projected_year5_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Primary Cause</span>'
            f'<span class="value">{deg.get("primary_cause", "-")}</span></div>\n'
            '</div>'
        )

    def _html_persistence_factors(self, data: Dict[str, Any]) -> str:
        """Render HTML persistence factors table."""
        factors = data.get("persistence_factors", [])
        rows = ""
        for f in factors:
            cls = "severity-low" if f.get("factor", 0) >= 0.8 else "severity-high"
            rows += (
                f'<tr><td>{f.get("ecm_name", "-")}</td>'
                f'<td>{self._fmt(f.get("year1_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(f.get("current_mwh", 0), 1)}</td>'
                f'<td class="{cls}">{self._fmt(f.get("factor", 0), 3)}</td>'
                f'<td>{f.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Persistence Factors</h2>\n'
            '<table>\n<tr><th>ECM</th><th>Year 1</th><th>Current</th>'
            f'<th>Factor</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_equipment_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML equipment performance table."""
        equipment = data.get("equipment_performance", [])
        rows = ""
        for eq in equipment:
            rows += (
                f'<tr><td>{eq.get("name", "-")}</td>'
                f'<td>{self._fmt(eq.get("design_efficiency", 0))}%</td>'
                f'<td>{self._fmt(eq.get("current_efficiency", 0))}%</td>'
                f'<td>{self._fmt(eq.get("decay_pct", 0))}%</td>'
                f'<td>{eq.get("maintenance_status", "-")}</td></tr>\n'
            )
        return (
            '<h2>5. Equipment Performance</h2>\n'
            '<table>\n<tr><th>Equipment</th><th>Design Eff.</th><th>Current Eff.</th>'
            f'<th>Decay</th><th>Maintenance</th></tr>\n{rows}</table>'
        )

    def _html_operational_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML operational changes table."""
        changes = data.get("operational_changes", [])
        rows = ""
        for c in changes:
            rows += (
                f'<tr><td>{c.get("date", "-")}</td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td>{self._fmt(c.get("impact_mwh", 0), 1)}</td>'
                f'<td>{c.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Operational Changes</h2>\n'
            '<table>\n<tr><th>Date</th><th>Change</th>'
            f'<th>Impact (MWh)</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_model_stability(self, data: Dict[str, Any]) -> str:
        """Render HTML model stability cards."""
        model = data.get("model_stability", {})
        return (
            '<h2>7. Model Stability</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value">{self._fmt(model.get("current_r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">CVRMSE</span>'
            f'<span class="value">{self._fmt(model.get("current_cvrmse", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">Coefficients</span>'
            f'<span class="value">{model.get("coefficient_stability", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Refit</span>'
            f'<span class="value">{model.get("refit_recommended", "-")}</span></div>\n'
            '</div>'
        )

    def _html_recommissioning(self, data: Dict[str, Any]) -> str:
        """Render HTML re-commissioning assessment."""
        recom = data.get("recommissioning", {})
        actions = recom.get("actions", [])
        rows = ""
        for a in actions:
            rows += (
                f'<tr><td>{a.get("action", "-")}</td>'
                f'<td>{a.get("ecm", "-")}</td>'
                f'<td>{a.get("priority", "-")}</td>'
                f'<td>{self._fmt(a.get("recovery_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(a.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>8. Re-commissioning Assessment</h2>\n'
            f'<p>Result: {recom.get("assessment_result", "-")} | '
            f'Recovery: {self._format_energy(recom.get("estimated_recovery_mwh", 0))} | '
            f'Cost: {self._format_currency(recom.get("estimated_cost", 0))}</p>\n'
            '<table>\n<tr><th>Action</th><th>ECM</th><th>Priority</th>'
            f'<th>Recovery (MWh)</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_risk_indicators(self, data: Dict[str, Any]) -> str:
        """Render HTML risk indicators table."""
        risks = data.get("risk_indicators", [])
        rows = ""
        for r in risks:
            cls = "severity-high" if r.get("status") == "ALERT" else "severity-low"
            rows += (
                f'<tr><td>{r.get("indicator", "-")}</td>'
                f'<td>{self._fmt(r.get("value", 0), 2)}</td>'
                f'<td>{self._fmt(r.get("threshold", 0), 2)}</td>'
                f'<td class="{cls}">{r.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Risk Indicators</h2>\n'
            '<table>\n<tr><th>Indicator</th><th>Value</th>'
            f'<th>Threshold</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Schedule re-commissioning for ECMs with persistence factor below 0.80",
            "Implement preventive maintenance to reduce equipment degradation",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_persistence_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON persistence summary."""
        s = data.get("persistence_summary", {})
        return {
            "year1_savings_mwh": s.get("year1_savings_mwh", 0),
            "current_savings_mwh": s.get("current_savings_mwh", 0),
            "persistence_factor": s.get("persistence_factor", 0),
            "degradation_rate_pct": s.get("degradation_rate_pct", 0),
            "cumulative_savings_mwh": s.get("cumulative_savings_mwh", 0),
            "ecms_at_risk": s.get("ecms_at_risk", 0),
            "overall_status": s.get("overall_status", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        yoy = data.get("yoy_savings", [])
        factors = data.get("persistence_factors", [])
        return {
            "savings_trend": {
                "type": "line",
                "labels": [y.get("year", "") for y in yoy],
                "values": [y.get("savings_mwh", 0) for y in yoy],
            },
            "persistence_by_ecm": {
                "type": "bar",
                "labels": [f.get("ecm_name", "") for f in factors],
                "values": [f.get("factor", 0) for f in factors],
            },
            "cumulative_savings": {
                "type": "area",
                "labels": [y.get("year", "") for y in yoy],
                "values": [y.get("cumulative_mwh", 0) for y in yoy],
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
