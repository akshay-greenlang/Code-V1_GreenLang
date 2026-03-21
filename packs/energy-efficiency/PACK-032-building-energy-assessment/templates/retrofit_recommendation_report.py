# -*- coding: utf-8 -*-
"""
RetrofitRecommendationReportTemplate - Retrofit business case report for PACK-032.

Generates retrofit recommendation reports with current performance
baselines, detailed retrofit measures, financial analysis including
NPV/IRR/payback, Marginal Abatement Cost Curve (MACC) data, staged
implementation roadmaps, funding/financing options, environmental
impact assessment, and risk analysis.

Sections:
    1. Executive Summary
    2. Current Performance Baseline
    3. Retrofit Measures Table
    4. Financial Analysis (NPV/IRR/Payback)
    5. MACC Curve Data
    6. Staged Roadmap
    7. Funding & Financing Options
    8. Environmental Impact
    9. Risk Assessment
   10. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RetrofitRecommendationReportTemplate:
    """
    Retrofit recommendation and business case report template.

    Renders retrofit reports with financial analysis, MACC curves,
    staged roadmaps, funding options, and risk assessments across
    markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RETROFIT_SECTIONS: List[str] = [
        "Executive Summary",
        "Current Performance",
        "Retrofit Measures",
        "Financial Analysis",
        "MACC Curve",
        "Staged Roadmap",
        "Funding Options",
        "Environmental Impact",
        "Risk Assessment",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RetrofitRecommendationReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render retrofit recommendation report as Markdown.

        Args:
            data: Retrofit analysis data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_current_performance(data),
            self._md_retrofit_measures(data),
            self._md_financial_analysis(data),
            self._md_macc_curve(data),
            self._md_staged_roadmap(data),
            self._md_funding_options(data),
            self._md_environmental_impact(data),
            self._md_risk_assessment(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render retrofit recommendation report as self-contained HTML.

        Args:
            data: Retrofit analysis data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_current_performance(data),
            self._html_retrofit_measures(data),
            self._html_financial_analysis(data),
            self._html_macc_curve(data),
            self._html_staged_roadmap(data),
            self._html_funding_options(data),
            self._html_environmental_impact(data),
            self._html_risk_assessment(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Retrofit Recommendation Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render retrofit recommendation report as structured JSON.

        Args:
            data: Retrofit analysis data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "retrofit_recommendation_report",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "current_performance": data.get("current_performance", {}),
            "retrofit_measures": data.get("retrofit_measures", []),
            "financial_analysis": data.get("financial_analysis", {}),
            "macc_curve": data.get("macc_curve", []),
            "staged_roadmap": data.get("staged_roadmap", []),
            "funding_options": data.get("funding_options", []),
            "environmental_impact": data.get("environmental_impact", {}),
            "risk_assessment": data.get("risk_assessment", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        building = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Retrofit Recommendation Report\n\n"
            f"**Building:** {building}  \n"
            f"**Address:** {data.get('address', '-')}  \n"
            f"**Analysis Date:** {data.get('analysis_date', '-')}  \n"
            f"**Prepared By:** {data.get('prepared_by', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 RetrofitRecommendationReportTemplate v32.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Current | Post-Retrofit | Improvement |\n"
            "|--------|---------|---------------|-------------|\n"
            f"| EUI (kWh/m2/yr) | {self._fmt(s.get('current_eui', 0))} "
            f"| {self._fmt(s.get('post_retrofit_eui', 0))} "
            f"| {self._fmt(s.get('eui_improvement_pct', 0))}% |\n"
            f"| CO2 (kgCO2/m2/yr) | {self._fmt(s.get('current_co2', 0))} "
            f"| {self._fmt(s.get('post_retrofit_co2', 0))} "
            f"| {self._fmt(s.get('co2_reduction_pct', 0))}% |\n"
            f"| Annual Cost | {s.get('current_cost', '-')} "
            f"| {s.get('post_retrofit_cost', '-')} "
            f"| {s.get('cost_saving', '-')} |\n"
            f"| EPC Rating | {s.get('current_epc', '-')} "
            f"| {s.get('post_retrofit_epc', '-')} "
            f"| {s.get('epc_improvement', '-')} |\n\n"
            f"**Total Investment:** {s.get('total_investment', '-')}  \n"
            f"**Total Measures:** {s.get('total_measures', 0)}  \n"
            f"**Portfolio NPV:** {s.get('portfolio_npv', '-')}  \n"
            f"**Blended Payback:** {self._fmt(s.get('blended_payback_years', 0), 1)} years"
        )

    def _md_current_performance(self, data: Dict[str, Any]) -> str:
        """Render current performance baseline section."""
        perf = data.get("current_performance", {})
        breakdowns = perf.get("end_use_breakdown", [])
        lines = [
            "## 2. Current Performance Baseline\n",
            f"**EUI:** {self._fmt(perf.get('eui_kwh_m2', 0))} kWh/m2/yr  ",
            f"**Annual Energy Cost:** {perf.get('annual_cost', '-')}  ",
            f"**Annual CO2:** {self._fmt(perf.get('annual_co2_tonnes', 0))} tonnes  ",
            f"**EPC Rating:** {perf.get('epc_rating', '-')}  ",
            f"**DEC Rating:** {perf.get('dec_rating', '-')}",
        ]
        if breakdowns:
            lines.extend([
                "\n### End-Use Breakdown\n",
                "| End Use | kWh/yr | Share (%) | Cost/yr |",
                "|---------|--------|-----------|---------|",
            ])
            for b in breakdowns:
                lines.append(
                    f"| {b.get('end_use', '-')} "
                    f"| {self._fmt(b.get('kwh_yr', 0), 0)} "
                    f"| {self._fmt(b.get('share_pct', 0))}% "
                    f"| {b.get('cost_yr', '-')} |"
                )
        return "\n".join(lines)

    def _md_retrofit_measures(self, data: Dict[str, Any]) -> str:
        """Render retrofit measures table."""
        measures = data.get("retrofit_measures", [])
        if not measures:
            return "## 3. Retrofit Measures\n\n_No retrofit measures identified._"
        lines = [
            "## 3. Retrofit Measures\n",
            "| # | Measure | Category | Savings (kWh/yr) | Cost | Payback (yr) | CO2 Saved |",
            "|---|---------|----------|-----------------|------|-------------|-----------|",
        ]
        for i, m in enumerate(measures, 1):
            lines.append(
                f"| {i} | {m.get('measure', '-')} "
                f"| {m.get('category', '-')} "
                f"| {self._fmt(m.get('savings_kwh', 0), 0)} "
                f"| {m.get('cost', '-')} "
                f"| {self._fmt(m.get('payback_years', 0), 1)} "
                f"| {m.get('co2_savings', '-')} |"
            )
        return "\n".join(lines)

    def _md_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render financial analysis section with NPV/IRR/payback."""
        fin = data.get("financial_analysis", {})
        measures_fin = fin.get("measures", [])
        lines = [
            "## 4. Financial Analysis\n",
            f"**Discount Rate:** {self._fmt(fin.get('discount_rate_pct', 0))}%  ",
            f"**Analysis Period:** {fin.get('analysis_period_years', 0)} years  ",
            f"**Energy Price Escalation:** {self._fmt(fin.get('energy_escalation_pct', 0))}%/yr  ",
            f"**Carbon Price Assumed:** {fin.get('carbon_price', '-')}/tCO2\n",
            "### Portfolio Summary\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Capital Cost | {fin.get('total_capex', '-')} |\n"
            f"| Total Annual Savings | {fin.get('total_annual_savings', '-')} |\n"
            f"| Portfolio NPV | {fin.get('npv', '-')} |\n"
            f"| Portfolio IRR | {self._fmt(fin.get('irr_pct', 0))}% |\n"
            f"| Simple Payback | {self._fmt(fin.get('simple_payback_years', 0), 1)} years |\n"
            f"| Discounted Payback | {self._fmt(fin.get('discounted_payback_years', 0), 1)} years |",
        ]
        if measures_fin:
            lines.extend([
                "\n### Measure-Level Financial Analysis\n",
                "| Measure | Capex | Annual Saving | NPV | IRR (%) | Payback (yr) |",
                "|---------|-------|--------------|-----|---------|-------------|",
            ])
            for m in measures_fin:
                lines.append(
                    f"| {m.get('measure', '-')} "
                    f"| {m.get('capex', '-')} "
                    f"| {m.get('annual_saving', '-')} "
                    f"| {m.get('npv', '-')} "
                    f"| {self._fmt(m.get('irr_pct', 0))} "
                    f"| {self._fmt(m.get('payback_years', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_macc_curve(self, data: Dict[str, Any]) -> str:
        """Render MACC (Marginal Abatement Cost Curve) data section."""
        macc = data.get("macc_curve", [])
        if not macc:
            return "## 5. Marginal Abatement Cost Curve\n\n_No MACC data available._"
        lines = [
            "## 5. Marginal Abatement Cost Curve\n",
            "The MACC ranks measures by cost per tonne of CO2 abated.\n",
            "| Measure | Abatement (tCO2/yr) | Cost/tCO2 | Cumulative Abatement |",
            "|---------|--------------------|-----------|--------------------|",
        ]
        cumulative = 0.0
        for m in macc:
            abatement = m.get("abatement_tco2", 0)
            cumulative += abatement
            lines.append(
                f"| {m.get('measure', '-')} "
                f"| {self._fmt(abatement, 1)} "
                f"| {m.get('cost_per_tco2', '-')} "
                f"| {self._fmt(cumulative, 1)} |"
            )
        return "\n".join(lines)

    def _md_staged_roadmap(self, data: Dict[str, Any]) -> str:
        """Render staged implementation roadmap."""
        stages = data.get("staged_roadmap", [])
        if not stages:
            return "## 6. Staged Implementation Roadmap\n\n_No roadmap defined._"
        lines = ["## 6. Staged Implementation Roadmap\n"]
        for stage in stages:
            lines.extend([
                f"### {stage.get('stage', 'Stage')} - {stage.get('timeframe', '-')}\n",
                f"**Budget:** {stage.get('budget', '-')}  ",
                f"**Expected EUI After:** {self._fmt(stage.get('eui_after', 0))} kWh/m2/yr  ",
                f"**Expected EPC After:** {stage.get('epc_after', '-')}\n",
                "| Measure | Priority | Dependencies |",
                "|---------|----------|-------------|",
            ])
            for m in stage.get("measures", []):
                lines.append(
                    f"| {m.get('measure', '-')} "
                    f"| {m.get('priority', '-')} "
                    f"| {m.get('dependencies', '-')} |"
                )
            lines.append("")
        return "\n".join(lines)

    def _md_funding_options(self, data: Dict[str, Any]) -> str:
        """Render funding and financing options section."""
        options = data.get("funding_options", [])
        if not options:
            return "## 7. Funding & Financing Options\n\n_No funding options identified._"
        lines = [
            "## 7. Funding & Financing Options\n",
            "| Option | Type | Amount | Term | Rate | Eligibility |",
            "|--------|------|--------|------|------|-------------|",
        ]
        for o in options:
            lines.append(
                f"| {o.get('name', '-')} "
                f"| {o.get('type', '-')} "
                f"| {o.get('amount', '-')} "
                f"| {o.get('term', '-')} "
                f"| {o.get('rate', '-')} "
                f"| {o.get('eligibility', '-')} |"
            )
        return "\n".join(lines)

    def _md_environmental_impact(self, data: Dict[str, Any]) -> str:
        """Render environmental impact section."""
        env = data.get("environmental_impact", {})
        return (
            "## 8. Environmental Impact\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Annual CO2 Reduction | {self._fmt(env.get('annual_co2_reduction_tonnes', 0))} tonnes |\n"
            f"| Lifetime CO2 Reduction | {self._fmt(env.get('lifetime_co2_reduction_tonnes', 0))} tonnes |\n"
            f"| CO2 Reduction (%) | {self._fmt(env.get('co2_reduction_pct', 0))}% |\n"
            f"| Equivalent Trees | {self._fmt(env.get('equivalent_trees', 0), 0)} |\n"
            f"| Equivalent Cars Removed | {self._fmt(env.get('equivalent_cars', 0), 0)} |\n"
            f"| Energy Saved (kWh/yr) | {self._fmt(env.get('energy_saved_kwh', 0), 0)} |\n"
            f"| Water Savings (m3/yr) | {self._fmt(env.get('water_savings_m3', 0), 0)} |\n"
            f"| Air Quality Improvement | {env.get('air_quality_impact', '-')} |"
        )

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        risks = data.get("risk_assessment", [])
        if not risks:
            return "## 9. Risk Assessment\n\n_No risks identified._"
        lines = [
            "## 9. Risk Assessment\n",
            "| Risk | Likelihood | Impact | Mitigation | Residual Risk |",
            "|------|-----------|--------|------------|---------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} "
                f"| {r.get('residual', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 RetrofitRecommendationReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        building = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Retrofit Recommendation Report</h1>\n'
            f'<p class="subtitle">Building: {building} | Generated: {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary with KPI cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">Investment</span>'
            f'<span class="value">{s.get("total_investment", "-")}</span></div>\n'
            f'<div class="card"><span class="label">NPV</span>'
            f'<span class="value">{s.get("portfolio_npv", "-")}</span></div>\n'
            f'<div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(s.get("blended_payback_years", 0), 1)} yr</span></div>\n'
            f'<div class="card"><span class="label">CO2 Cut</span>'
            f'<span class="value">{self._fmt(s.get("co2_reduction_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_current_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML current performance."""
        perf = data.get("current_performance", {})
        breakdown = perf.get("end_use_breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("end_use", "-")}</td>'
                f'<td>{self._fmt(b.get("kwh_yr", 0), 0)}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Current Performance Baseline</h2>\n'
            f'<p>EUI: {self._fmt(perf.get("eui_kwh_m2", 0))} kWh/m2/yr | '
            f'EPC: {perf.get("epc_rating", "-")}</p>\n'
            '<table>\n<tr><th>End Use</th><th>kWh/yr</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_retrofit_measures(self, data: Dict[str, Any]) -> str:
        """Render HTML retrofit measures table."""
        measures = data.get("retrofit_measures", [])
        rows = ""
        for i, m in enumerate(measures, 1):
            rows += (
                f'<tr><td>{i}</td><td>{m.get("measure", "-")}</td>'
                f'<td>{m.get("category", "-")}</td>'
                f'<td>{self._fmt(m.get("savings_kwh", 0), 0)}</td>'
                f'<td>{m.get("cost", "-")}</td>'
                f'<td>{self._fmt(m.get("payback_years", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Retrofit Measures</h2>\n'
            '<table>\n<tr><th>#</th><th>Measure</th><th>Category</th>'
            f'<th>Savings (kWh)</th><th>Cost</th><th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML financial analysis."""
        fin = data.get("financial_analysis", {})
        return (
            '<h2>Financial Analysis</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Total Capex</td><td>{fin.get("total_capex", "-")}</td></tr>\n'
            f'<tr><td>Annual Savings</td><td>{fin.get("total_annual_savings", "-")}</td></tr>\n'
            f'<tr><td>NPV</td><td>{fin.get("npv", "-")}</td></tr>\n'
            f'<tr><td>IRR</td><td>{self._fmt(fin.get("irr_pct", 0))}%</td></tr>\n'
            f'<tr><td>Simple Payback</td><td>{self._fmt(fin.get("simple_payback_years", 0), 1)} yr</td></tr>\n'
            '</table>'
        )

    def _html_macc_curve(self, data: Dict[str, Any]) -> str:
        """Render HTML MACC curve data table."""
        macc = data.get("macc_curve", [])
        rows = ""
        for m in macc:
            rows += (
                f'<tr><td>{m.get("measure", "-")}</td>'
                f'<td>{self._fmt(m.get("abatement_tco2", 0), 1)}</td>'
                f'<td>{m.get("cost_per_tco2", "-")}</td></tr>\n'
            )
        return (
            '<h2>Marginal Abatement Cost Curve</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Abatement (tCO2)</th>'
            f'<th>Cost/tCO2</th></tr>\n{rows}</table>'
        )

    def _html_staged_roadmap(self, data: Dict[str, Any]) -> str:
        """Render HTML staged roadmap."""
        stages = data.get("staged_roadmap", [])
        content = ""
        for stage in stages:
            measures = "".join(
                f'<li>{m.get("measure", "-")} ({m.get("priority", "-")})</li>'
                for m in stage.get("measures", [])
            )
            content += (
                f'<div class="phase"><h3>{stage.get("stage", "Stage")} - '
                f'{stage.get("timeframe", "-")}</h3>'
                f'<p>Budget: {stage.get("budget", "-")} | '
                f'EPC After: {stage.get("epc_after", "-")}</p>'
                f'<ul>{measures}</ul></div>\n'
            )
        return f'<h2>Staged Implementation Roadmap</h2>\n{content}'

    def _html_funding_options(self, data: Dict[str, Any]) -> str:
        """Render HTML funding options."""
        options = data.get("funding_options", [])
        rows = ""
        for o in options:
            rows += (
                f'<tr><td>{o.get("name", "-")}</td>'
                f'<td>{o.get("type", "-")}</td>'
                f'<td>{o.get("amount", "-")}</td>'
                f'<td>{o.get("eligibility", "-")}</td></tr>\n'
            )
        return (
            '<h2>Funding &amp; Financing Options</h2>\n'
            '<table>\n<tr><th>Option</th><th>Type</th><th>Amount</th>'
            f'<th>Eligibility</th></tr>\n{rows}</table>'
        )

    def _html_environmental_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML environmental impact."""
        env = data.get("environmental_impact", {})
        return (
            '<h2>Environmental Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">CO2/yr Saved</span>'
            f'<span class="value">{self._fmt(env.get("annual_co2_reduction_tonnes", 0))}</span>'
            f'<span class="label">tonnes</span></div>\n'
            f'<div class="card"><span class="label">Energy Saved</span>'
            f'<span class="value">{self._fmt(env.get("energy_saved_kwh", 0), 0)}</span>'
            f'<span class="label">kWh/yr</span></div>\n'
            f'<div class="card"><span class="label">Trees Equiv</span>'
            f'<span class="value">{self._fmt(env.get("equivalent_trees", 0), 0)}</span></div>\n'
            '</div>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment."""
        risks = data.get("risk_assessment", [])
        rows = ""
        for r in risks:
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td>{r.get("impact", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Risk Assessment</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Likelihood</th><th>Impact</th>'
            f'<th>Mitigation</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        s = data.get("executive_summary", {})
        return {
            "current_eui": s.get("current_eui", 0),
            "post_retrofit_eui": s.get("post_retrofit_eui", 0),
            "eui_improvement_pct": s.get("eui_improvement_pct", 0),
            "current_co2": s.get("current_co2", 0),
            "post_retrofit_co2": s.get("post_retrofit_co2", 0),
            "co2_reduction_pct": s.get("co2_reduction_pct", 0),
            "total_investment": s.get("total_investment", ""),
            "portfolio_npv": s.get("portfolio_npv", ""),
            "blended_payback_years": s.get("blended_payback_years", 0),
            "total_measures": s.get("total_measures", 0),
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
            "h3{color:#0d6efd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".phase{border-left:3px solid #0d6efd;padding-left:15px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
