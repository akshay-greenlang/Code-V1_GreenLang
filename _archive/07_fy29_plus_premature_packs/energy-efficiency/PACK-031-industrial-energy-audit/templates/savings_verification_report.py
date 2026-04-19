# -*- coding: utf-8 -*-
"""
SavingsVerificationReportTemplate - IPMVP-compliant M&V report for PACK-031.

Generates Measurement and Verification reports compliant with the
International Performance Measurement and Verification Protocol (IPMVP).
Covers all four IPMVP Options (A, B, C, D), baseline adjustments,
post-implementation measurements, verified savings with confidence intervals,
and cost savings verification.

Sections:
    1. Executive Summary
    2. M&V Plan Summary
    3. IPMVP Option & Boundary
    4. Baseline Data & Model
    5. Adjustments (Routine & Non-Routine)
    6. Post-Implementation Measurements
    7. Verified Savings
    8. Confidence & Uncertainty
    9. Cost Savings
    10. Conclusions & Recommendations

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SavingsVerificationReportTemplate:
    """
    IPMVP-compliant savings verification report template.

    Renders M&V reports with baseline data, adjustments, post-period
    measurements, verified energy savings, confidence intervals,
    and cost savings across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    IPMVP_OPTIONS: Dict[str, str] = {
        "A": "Retrofit Isolation - Key Parameter Measurement",
        "B": "Retrofit Isolation - All Parameter Measurement",
        "C": "Whole Facility",
        "D": "Calibrated Simulation",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SavingsVerificationReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render IPMVP savings verification report as Markdown.

        Args:
            data: Verification engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_mv_plan_summary(data),
            self._md_ipmvp_option(data),
            self._md_baseline_data(data),
            self._md_adjustments(data),
            self._md_post_implementation(data),
            self._md_verified_savings(data),
            self._md_confidence_uncertainty(data),
            self._md_cost_savings(data),
            self._md_conclusions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render IPMVP savings verification report as HTML.

        Args:
            data: Verification engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_baseline_data(data),
            self._html_verified_savings(data),
            self._html_confidence(data),
            self._html_cost_savings(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Savings Verification Report (IPMVP)</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render savings verification report as structured JSON.

        Args:
            data: Verification engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "savings_verification_report",
            "version": "31.0.0",
            "standard": "IPMVP 2022",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "mv_plan": data.get("mv_plan", {}),
            "ipmvp_option": data.get("ipmvp_option", {}),
            "baseline": data.get("baseline", {}),
            "adjustments": data.get("adjustments", {}),
            "post_implementation": data.get("post_implementation", {}),
            "verified_savings": data.get("verified_savings", {}),
            "confidence": data.get("confidence", {}),
            "cost_savings": data.get("cost_savings", {}),
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
        project = data.get("project_name", "Energy Conservation Measure")
        facility = data.get("facility_name", "Industrial Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Savings Verification Report (IPMVP)\n\n"
            f"**Project:** {project}  \n"
            f"**Facility:** {facility}  \n"
            f"**Standard:** IPMVP 2022 (EVO 10000-1:2022)  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 SavingsVerificationReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        vs = data.get("verified_savings", {})
        cost = data.get("cost_savings", {})
        conf = data.get("confidence", {})
        option = data.get("ipmvp_option", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| IPMVP Option | Option {option.get('option', 'C')}: "
            f"{self.IPMVP_OPTIONS.get(option.get('option', 'C'), '-')} |\n"
            f"| Verified Energy Savings | {self._fmt(vs.get('total_savings_mwh', 0))} MWh/yr |\n"
            f"| Savings as % of Baseline | {self._fmt(vs.get('savings_pct_of_baseline', 0))}% |\n"
            f"| Verified Cost Savings | EUR {self._fmt(cost.get('total_cost_savings_eur', 0))} /yr |\n"
            f"| Confidence Level | {self._fmt(conf.get('confidence_level_pct', 90))}% |\n"
            f"| Precision | +/- {self._fmt(conf.get('precision_pct', 0))}% |\n"
            f"| CO2 Reduction | {self._fmt(vs.get('co2_reduction_tonnes', 0))} tonnes/yr |"
        )

    def _md_mv_plan_summary(self, data: Dict[str, Any]) -> str:
        """Render M&V plan summary."""
        plan = data.get("mv_plan", {})
        return (
            "## 2. M&V Plan Summary\n\n"
            f"- **Plan Revision:** {plan.get('revision', '1.0')}\n"
            f"- **ECMs Covered:** {plan.get('ecm_count', 0)}\n"
            f"- **Measurement Boundary:** {plan.get('boundary', 'Whole Facility')}\n"
            f"- **Metering Equipment:** {plan.get('metering', '-')}\n"
            f"- **Data Collection Interval:** {plan.get('interval', 'Monthly')}\n"
            f"- **Baseline Period:** {plan.get('baseline_period', '-')}\n"
            f"- **Post-Period:** {plan.get('post_period', '-')}\n"
            f"- **Verification Frequency:** {plan.get('verification_frequency', 'Annual')}"
        )

    def _md_ipmvp_option(self, data: Dict[str, Any]) -> str:
        """Render IPMVP option selection and justification."""
        option = data.get("ipmvp_option", {})
        opt = option.get("option", "C")
        lines = [
            "## 3. IPMVP Option & Measurement Boundary\n",
            f"**Selected Option:** Option {opt} - {self.IPMVP_OPTIONS.get(opt, '-')}  ",
            f"**Justification:** {option.get('justification', '-')}  ",
            f"**Measurement Boundary:** {option.get('boundary_description', '-')}  ",
        ]
        ecms = option.get("ecms", [])
        if ecms:
            lines.extend([
                "\n### Energy Conservation Measures (ECMs)\n",
                "| # | ECM | Description | Expected Savings (MWh) | Status |",
                "|---|-----|-------------|----------------------|--------|",
            ])
            for i, ecm in enumerate(ecms, 1):
                lines.append(
                    f"| {i} | {ecm.get('name', '-')} "
                    f"| {ecm.get('description', '-')} "
                    f"| {self._fmt(ecm.get('expected_savings_mwh', 0))} "
                    f"| {ecm.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_baseline_data(self, data: Dict[str, Any]) -> str:
        """Render baseline data section."""
        baseline = data.get("baseline", {})
        model = baseline.get("model", {})
        lines = [
            "## 4. Baseline Data & Model\n",
            f"**Baseline Period:** {baseline.get('period', '-')}  ",
            f"**Total Baseline Consumption:** {self._fmt(baseline.get('total_mwh', 0))} MWh  ",
            f"**Model Type:** {model.get('type', 'Linear Regression')}  ",
            f"**R-Squared:** {self._fmt(model.get('r_squared', 0), 4)}  ",
            f"**CV(RMSE):** {self._fmt(model.get('cv_rmse_pct', 0), 2)}%  ",
            f"**Equation:** {model.get('equation', '-')}",
        ]
        return "\n".join(lines)

    def _md_adjustments(self, data: Dict[str, Any]) -> str:
        """Render routine and non-routine adjustments."""
        adj = data.get("adjustments", {})
        routine = adj.get("routine", [])
        non_routine = adj.get("non_routine", [])
        lines = ["## 5. Adjustments\n"]
        lines.append("### Routine Adjustments\n")
        if routine:
            lines.extend([
                "| Variable | Baseline Value | Reporting Value | Adjustment (MWh) |",
                "|----------|---------------|-----------------|------------------|",
            ])
            for r in routine:
                lines.append(
                    f"| {r.get('variable', '-')} "
                    f"| {self._fmt(r.get('baseline_value', 0))} "
                    f"| {self._fmt(r.get('reporting_value', 0))} "
                    f"| {self._fmt(r.get('adjustment_mwh', 0))} |"
                )
        else:
            lines.append("_No routine adjustments required._")
        lines.append("\n### Non-Routine Adjustments\n")
        if non_routine:
            lines.extend([
                "| Event | Description | Period | Adjustment (MWh) |",
                "|-------|-------------|--------|------------------|",
            ])
            for nr in non_routine:
                lines.append(
                    f"| {nr.get('event', '-')} "
                    f"| {nr.get('description', '-')} "
                    f"| {nr.get('period', '-')} "
                    f"| {self._fmt(nr.get('adjustment_mwh', 0))} |"
                )
        else:
            lines.append("_No non-routine adjustments required._")
        total_adj = adj.get("total_adjustment_mwh", 0)
        lines.append(f"\n**Total Adjustments:** {self._fmt(total_adj)} MWh")
        return "\n".join(lines)

    def _md_post_implementation(self, data: Dict[str, Any]) -> str:
        """Render post-implementation measurements."""
        post = data.get("post_implementation", {})
        monthly = post.get("monthly_data", [])
        lines = [
            "## 6. Post-Implementation Measurements\n",
            f"**Post Period:** {post.get('period', '-')}  ",
            f"**Total Measured Consumption:** {self._fmt(post.get('total_measured_mwh', 0))} MWh  ",
            f"**Adjusted Baseline (for post conditions):** "
            f"{self._fmt(post.get('adjusted_baseline_mwh', 0))} MWh",
        ]
        if monthly:
            lines.extend([
                "\n| Month | Measured (MWh) | Adj. Baseline (MWh) | Savings (MWh) |",
                "|-------|---------------|--------------------|--------------:|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('measured_mwh', 0))} "
                    f"| {self._fmt(m.get('adjusted_baseline_mwh', 0))} "
                    f"| {self._fmt(m.get('savings_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_verified_savings(self, data: Dict[str, Any]) -> str:
        """Render verified savings section."""
        vs = data.get("verified_savings", {})
        by_ecm = vs.get("by_ecm", [])
        lines = [
            "## 7. Verified Savings\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Adjusted Baseline Consumption | {self._fmt(vs.get('adjusted_baseline_mwh', 0))} MWh |",
            f"| Post-Period Consumption | {self._fmt(vs.get('post_period_mwh', 0))} MWh |",
            f"| **Verified Energy Savings** | **{self._fmt(vs.get('total_savings_mwh', 0))} MWh** |",
            f"| Savings % of Baseline | {self._fmt(vs.get('savings_pct_of_baseline', 0))}% |",
            f"| CO2 Reduction | {self._fmt(vs.get('co2_reduction_tonnes', 0))} tonnes |",
        ]
        if by_ecm:
            lines.extend([
                "\n### Savings by ECM\n",
                "| ECM | Expected (MWh) | Verified (MWh) | Achievement (%) |",
                "|-----|---------------|----------------|-----------------|",
            ])
            for ecm in by_ecm:
                lines.append(
                    f"| {ecm.get('name', '-')} "
                    f"| {self._fmt(ecm.get('expected_mwh', 0))} "
                    f"| {self._fmt(ecm.get('verified_mwh', 0))} "
                    f"| {self._fmt(ecm.get('achievement_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_confidence_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render confidence and uncertainty analysis."""
        conf = data.get("confidence", {})
        sources = conf.get("uncertainty_sources", [])
        lines = [
            "## 8. Confidence & Uncertainty Analysis\n",
            f"**Confidence Level:** {self._fmt(conf.get('confidence_level_pct', 90))}%  ",
            f"**Precision:** +/- {self._fmt(conf.get('precision_pct', 0))}%  ",
            f"**Savings Range:** {self._fmt(conf.get('lower_bound_mwh', 0))} to "
            f"{self._fmt(conf.get('upper_bound_mwh', 0))} MWh  ",
            f"**Model Uncertainty:** {self._fmt(conf.get('model_uncertainty_pct', 0))}%  ",
            f"**Measurement Uncertainty:** {self._fmt(conf.get('measurement_uncertainty_pct', 0))}%  ",
            f"**Combined Uncertainty:** {self._fmt(conf.get('combined_uncertainty_pct', 0))}%",
        ]
        if sources:
            lines.extend([
                "\n### Uncertainty Sources\n",
                "| Source | Type | Contribution (%) |",
                "|--------|------|-----------------|",
            ])
            for s in sources:
                lines.append(
                    f"| {s.get('source', '-')} "
                    f"| {s.get('type', '-')} "
                    f"| {self._fmt(s.get('contribution_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_cost_savings(self, data: Dict[str, Any]) -> str:
        """Render cost savings section."""
        cost = data.get("cost_savings", {})
        by_source = cost.get("by_energy_source", [])
        lines = [
            "## 9. Cost Savings\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Verified Cost Savings | EUR {self._fmt(cost.get('total_cost_savings_eur', 0))} /yr |",
            f"| Energy Cost Savings | EUR {self._fmt(cost.get('energy_cost_savings_eur', 0))} /yr |",
            f"| Demand Cost Savings | EUR {self._fmt(cost.get('demand_cost_savings_eur', 0))} /yr |",
            f"| O&M Cost Savings | EUR {self._fmt(cost.get('om_cost_savings_eur', 0))} /yr |",
            f"| Project Investment | EUR {self._fmt(cost.get('investment_eur', 0))} |",
            f"| Simple Payback | {self._fmt(cost.get('simple_payback_years', 0), 1)} years |",
            f"| NPV (10 yr) | EUR {self._fmt(cost.get('npv_eur', 0))} |",
            f"| IRR | {self._fmt(cost.get('irr_pct', 0))}% |",
        ]
        if by_source:
            lines.extend([
                "\n### Cost Savings by Energy Source\n",
                "| Source | Savings (MWh) | Unit Rate (EUR/MWh) | Cost Savings (EUR) |",
                "|--------|--------------|--------------------|--------------------|",
            ])
            for s in by_source:
                lines.append(
                    f"| {s.get('source', '-')} "
                    f"| {self._fmt(s.get('savings_mwh', 0))} "
                    f"| {self._fmt(s.get('unit_rate_eur', 0))} "
                    f"| {self._fmt(s.get('cost_savings_eur', 0))} |"
                )
        return "\n".join(lines)

    def _md_conclusions(self, data: Dict[str, Any]) -> str:
        """Render conclusions and recommendations."""
        conclusions = data.get("conclusions", [])
        recs = data.get("mv_recommendations", [])
        lines = ["## 10. Conclusions & Recommendations\n"]
        lines.append("### Conclusions\n")
        if conclusions:
            for c in conclusions:
                lines.append(f"- {c}")
        else:
            lines.append("- Savings verification completed per IPMVP protocol")
        lines.append("\n### M&V Recommendations\n")
        if recs:
            for r in recs:
                lines.append(f"- {r}")
        else:
            lines.append("- Continue monitoring and annual verification")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        project = data.get("project_name", "Energy Conservation Measure")
        return (
            f'<h1>Savings Verification Report (IPMVP)</h1>\n'
            f'<p class="subtitle">Project: {project} | Standard: IPMVP 2022</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        vs = data.get("verified_savings", {})
        cost = data.get("cost_savings", {})
        conf = data.get("confidence", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Verified Savings</span>'
            f'<span class="value">{self._fmt(vs.get("total_savings_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">EUR {self._fmt(cost.get("total_cost_savings_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Confidence</span>'
            f'<span class="value">{self._fmt(conf.get("confidence_level_pct", 90))}%</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Reduction</span>'
            f'<span class="value">{self._fmt(vs.get("co2_reduction_tonnes", 0))} t</span></div>\n'
            '</div>'
        )

    def _html_baseline_data(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline data."""
        baseline = data.get("baseline", {})
        return (
            '<h2>Baseline Data</h2>\n'
            f'<p>Period: {baseline.get("period", "-")} | '
            f'Consumption: {self._fmt(baseline.get("total_mwh", 0))} MWh</p>'
        )

    def _html_verified_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML verified savings."""
        vs = data.get("verified_savings", {})
        return (
            '<h2>Verified Savings</h2>\n'
            f'<div class="savings-highlight">'
            f'<span class="big-number">{self._fmt(vs.get("total_savings_mwh", 0))}</span> MWh/yr '
            f'({self._fmt(vs.get("savings_pct_of_baseline", 0))}% of baseline)</div>'
        )

    def _html_confidence(self, data: Dict[str, Any]) -> str:
        """Render HTML confidence interval."""
        conf = data.get("confidence", {})
        return (
            '<h2>Confidence & Uncertainty</h2>\n'
            f'<p>{self._fmt(conf.get("confidence_level_pct", 90))}% confidence: '
            f'{self._fmt(conf.get("lower_bound_mwh", 0))} - '
            f'{self._fmt(conf.get("upper_bound_mwh", 0))} MWh</p>'
        )

    def _html_cost_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML cost savings."""
        cost = data.get("cost_savings", {})
        return (
            '<h2>Cost Savings</h2>\n'
            f'<p>Total: EUR {self._fmt(cost.get("total_cost_savings_eur", 0))}/yr | '
            f'Payback: {self._fmt(cost.get("simple_payback_years", 0), 1)} years | '
            f'IRR: {self._fmt(cost.get("irr_pct", 0))}%</p>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        vs = data.get("verified_savings", {})
        cost = data.get("cost_savings", {})
        conf = data.get("confidence", {})
        return {
            "ipmvp_option": data.get("ipmvp_option", {}).get("option", "C"),
            "total_savings_mwh": vs.get("total_savings_mwh", 0),
            "savings_pct_of_baseline": vs.get("savings_pct_of_baseline", 0),
            "total_cost_savings_eur": cost.get("total_cost_savings_eur", 0),
            "confidence_level_pct": conf.get("confidence_level_pct", 90),
            "precision_pct": conf.get("precision_pct", 0),
            "co2_reduction_tonnes": vs.get("co2_reduction_tonnes", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        post = data.get("post_implementation", {}).get("monthly_data", [])
        by_ecm = data.get("verified_savings", {}).get("by_ecm", [])
        return {
            "savings_timeline": {
                "type": "line",
                "labels": [m.get("month", "") for m in post],
                "series": {
                    "measured": [m.get("measured_mwh", 0) for m in post],
                    "adjusted_baseline": [m.get("adjusted_baseline_mwh", 0) for m in post],
                    "savings": [m.get("savings_mwh", 0) for m in post],
                },
            },
            "ecm_comparison": {
                "type": "grouped_bar",
                "labels": [e.get("name", "") for e in by_ecm],
                "series": {
                    "expected": [e.get("expected_mwh", 0) for e in by_ecm],
                    "verified": [e.get("verified_mwh", 0) for e in by_ecm],
                },
            },
            "confidence_interval": {
                "type": "error_bar",
                "central": data.get("verified_savings", {}).get("total_savings_mwh", 0),
                "lower": data.get("confidence", {}).get("lower_bound_mwh", 0),
                "upper": data.get("confidence", {}).get("upper_bound_mwh", 0),
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".savings-highlight{background:#d1e7dd;border-radius:8px;padding:20px;text-align:center;margin:15px 0;}"
            ".big-number{font-size:2.5em;font-weight:700;color:#0f5132;}"
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
