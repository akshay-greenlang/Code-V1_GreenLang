# -*- coding: utf-8 -*-
"""
OptionComparisonReportTemplate - IPMVP Option Comparison Report for PACK-040.

Generates comprehensive IPMVP option comparison reports covering
suitability scoring for Options A/B/C/D, cost-effectiveness analysis,
accuracy trade-off evaluation, ECM-specific recommendations, and
decision rationale documentation.

Sections:
    1. Comparison Overview
    2. ECM Characteristics
    3. Option A Assessment
    4. Option B Assessment
    5. Option C Assessment
    6. Option D Assessment
    7. Suitability Scoring
    8. Cost-Effectiveness
    9. Accuracy Trade-offs
    10. Recommendation

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (Options A/B/C/D)
    - ASHRAE Guideline 14-2014
    - ISO 50015:2014 (M&V option selection)
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


class OptionComparisonReportTemplate:
    """
    IPMVP option comparison report template.

    Renders comprehensive IPMVP option comparison reports showing
    suitability scores for Options A through D, cost-effectiveness
    analysis, accuracy-versus-cost trade-offs, ECM characteristics
    driving option selection, and final recommendation with rationale
    across markdown, HTML, and JSON formats. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize OptionComparisonReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render IPMVP option comparison report as Markdown.

        Args:
            data: IPMVP option engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_comparison_overview(data),
            self._md_ecm_characteristics(data),
            self._md_option_a(data),
            self._md_option_b(data),
            self._md_option_c(data),
            self._md_option_d(data),
            self._md_suitability_scoring(data),
            self._md_cost_effectiveness(data),
            self._md_accuracy_tradeoffs(data),
            self._md_recommendation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render IPMVP option comparison report as self-contained HTML.

        Args:
            data: IPMVP option engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_comparison_overview(data),
            self._html_ecm_characteristics(data),
            self._html_option_a(data),
            self._html_option_b(data),
            self._html_option_c(data),
            self._html_option_d(data),
            self._html_suitability_scoring(data),
            self._html_cost_effectiveness(data),
            self._html_accuracy_tradeoffs(data),
            self._html_recommendation(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>IPMVP Option Comparison Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render IPMVP option comparison report as structured JSON.

        Args:
            data: IPMVP option engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "option_comparison_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "comparison_overview": self._json_overview(data),
            "ecm_characteristics": data.get("ecm_characteristics", {}),
            "option_a": data.get("option_a", {}),
            "option_b": data.get("option_b", {}),
            "option_c": data.get("option_c", {}),
            "option_d": data.get("option_d", {}),
            "suitability_scoring": data.get("suitability_scoring", []),
            "cost_effectiveness": data.get("cost_effectiveness", []),
            "accuracy_tradeoffs": data.get("accuracy_tradeoffs", []),
            "recommendation": data.get("recommendation", {}),
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
            f"# IPMVP Option Comparison Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**ECM:** {data.get('ecm_name', '-')}  \n"
            f"**Estimated Savings:** {self._format_energy(data.get('estimated_savings_mwh', 0))}  \n"
            f"**Recommended Option:** {data.get('recommended_option', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 OptionComparisonReportTemplate v40.0.0\n\n---"
        )

    def _md_comparison_overview(self, data: Dict[str, Any]) -> str:
        """Render comparison overview section."""
        overview = data.get("comparison_overview", {})
        return (
            "## 1. Comparison Overview\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| ECM Type | {overview.get('ecm_type', '-')} |\n"
            f"| Estimated Savings | {self._format_energy(overview.get('estimated_savings_mwh', 0))} |\n"
            f"| Savings Fraction | {self._fmt(overview.get('savings_fraction_pct', 0))}% |\n"
            f"| ECM Interaction | {overview.get('ecm_interaction', '-')} |\n"
            f"| Isolation Possible | {overview.get('isolation_possible', '-')} |\n"
            f"| Metering Available | {overview.get('metering_available', '-')} |\n"
            f"| Budget for M&V | {self._format_currency(overview.get('mv_budget', 0))} |\n"
            f"| Required Accuracy | {self._fmt(overview.get('required_accuracy_pct', 0))}% |"
        )

    def _md_ecm_characteristics(self, data: Dict[str, Any]) -> str:
        """Render ECM characteristics section."""
        ecm = data.get("ecm_characteristics", {})
        if not ecm:
            return "## 2. ECM Characteristics\n\n_No ECM characteristics data available._"
        return (
            "## 2. ECM Characteristics\n\n"
            f"**Description:** {ecm.get('description', '-')}  \n"
            f"**Technology:** {ecm.get('technology', '-')}  \n"
            f"**End Use:** {ecm.get('end_use', '-')}  \n"
            f"**Affected Systems:** {', '.join(ecm.get('affected_systems', []))}  \n"
            f"**Installation Date:** {ecm.get('installation_date', '-')}  \n"
            f"**Expected Life:** {ecm.get('expected_life_years', '-')} years  \n"
            f"**Capital Cost:** {self._format_currency(ecm.get('capital_cost', 0))}  \n"
            f"**Interactive Effects:** {ecm.get('interactive_effects', 'None')}"
        )

    def _md_option_a(self, data: Dict[str, Any]) -> str:
        """Render Option A assessment section."""
        opt = data.get("option_a", {})
        if not opt:
            return "## 3. Option A - Retrofit Isolation: Key Parameter\n\n_No Option A data available._"
        return (
            "## 3. Option A - Retrofit Isolation: Key Parameter\n\n"
            "| Criterion | Assessment |\n|-----------|------------|\n"
            f"| Applicability | {opt.get('applicability', '-')} |\n"
            f"| Key Parameter | {opt.get('key_parameter', '-')} |\n"
            f"| Stipulated Values | {opt.get('stipulated_values', '-')} |\n"
            f"| Measurement Duration | {opt.get('measurement_duration', '-')} |\n"
            f"| Expected Accuracy | {self._fmt(opt.get('expected_accuracy_pct', 0))}% |\n"
            f"| M&V Cost | {self._format_currency(opt.get('mv_cost', 0))} |\n"
            f"| Pros | {opt.get('pros', '-')} |\n"
            f"| Cons | {opt.get('cons', '-')} |\n"
            f"| Suitability Score | {self._fmt(opt.get('suitability_score', 0), 1)}/10 |"
        )

    def _md_option_b(self, data: Dict[str, Any]) -> str:
        """Render Option B assessment section."""
        opt = data.get("option_b", {})
        if not opt:
            return "## 4. Option B - Retrofit Isolation: All Parameters\n\n_No Option B data available._"
        return (
            "## 4. Option B - Retrofit Isolation: All Parameters\n\n"
            "| Criterion | Assessment |\n|-----------|------------|\n"
            f"| Applicability | {opt.get('applicability', '-')} |\n"
            f"| Measured Parameters | {opt.get('measured_parameters', '-')} |\n"
            f"| Metering Required | {opt.get('metering_required', '-')} |\n"
            f"| Measurement Duration | {opt.get('measurement_duration', '-')} |\n"
            f"| Expected Accuracy | {self._fmt(opt.get('expected_accuracy_pct', 0))}% |\n"
            f"| M&V Cost | {self._format_currency(opt.get('mv_cost', 0))} |\n"
            f"| Pros | {opt.get('pros', '-')} |\n"
            f"| Cons | {opt.get('cons', '-')} |\n"
            f"| Suitability Score | {self._fmt(opt.get('suitability_score', 0), 1)}/10 |"
        )

    def _md_option_c(self, data: Dict[str, Any]) -> str:
        """Render Option C assessment section."""
        opt = data.get("option_c", {})
        if not opt:
            return "## 5. Option C - Whole Facility\n\n_No Option C data available._"
        return (
            "## 5. Option C - Whole Facility\n\n"
            "| Criterion | Assessment |\n|-----------|------------|\n"
            f"| Applicability | {opt.get('applicability', '-')} |\n"
            f"| Regression Model | {opt.get('regression_model', '-')} |\n"
            f"| Independent Variables | {opt.get('independent_variables', '-')} |\n"
            f"| Data Requirements | {opt.get('data_requirements', '-')} |\n"
            f"| Expected Accuracy | {self._fmt(opt.get('expected_accuracy_pct', 0))}% |\n"
            f"| M&V Cost | {self._format_currency(opt.get('mv_cost', 0))} |\n"
            f"| Pros | {opt.get('pros', '-')} |\n"
            f"| Cons | {opt.get('cons', '-')} |\n"
            f"| Suitability Score | {self._fmt(opt.get('suitability_score', 0), 1)}/10 |"
        )

    def _md_option_d(self, data: Dict[str, Any]) -> str:
        """Render Option D assessment section."""
        opt = data.get("option_d", {})
        if not opt:
            return "## 6. Option D - Calibrated Simulation\n\n_No Option D data available._"
        return (
            "## 6. Option D - Calibrated Simulation\n\n"
            "| Criterion | Assessment |\n|-----------|------------|\n"
            f"| Applicability | {opt.get('applicability', '-')} |\n"
            f"| Simulation Tool | {opt.get('simulation_tool', '-')} |\n"
            f"| Calibration Method | {opt.get('calibration_method', '-')} |\n"
            f"| Calibration Criteria | CVRMSE <= {self._fmt(opt.get('cvrmse_threshold', 15), 0)}% |\n"
            f"| Expected Accuracy | {self._fmt(opt.get('expected_accuracy_pct', 0))}% |\n"
            f"| M&V Cost | {self._format_currency(opt.get('mv_cost', 0))} |\n"
            f"| Pros | {opt.get('pros', '-')} |\n"
            f"| Cons | {opt.get('cons', '-')} |\n"
            f"| Suitability Score | {self._fmt(opt.get('suitability_score', 0), 1)}/10 |"
        )

    def _md_suitability_scoring(self, data: Dict[str, Any]) -> str:
        """Render suitability scoring section."""
        scores = data.get("suitability_scoring", [])
        if not scores:
            return "## 7. Suitability Scoring\n\n_No suitability scoring data available._"
        lines = [
            "## 7. Suitability Scoring\n",
            "| Criterion | Weight | Opt A | Opt B | Opt C | Opt D |",
            "|-----------|------:|-----:|-----:|-----:|-----:|",
        ]
        for s in scores:
            lines.append(
                f"| {s.get('criterion', '-')} "
                f"| {self._fmt(s.get('weight', 0), 1)} "
                f"| {self._fmt(s.get('option_a', 0), 1)} "
                f"| {self._fmt(s.get('option_b', 0), 1)} "
                f"| {self._fmt(s.get('option_c', 0), 1)} "
                f"| {self._fmt(s.get('option_d', 0), 1)} |"
            )
        totals = data.get("suitability_totals", {})
        if totals:
            lines.append(
                f"| **Weighted Total** | - "
                f"| **{self._fmt(totals.get('option_a', 0), 1)}** "
                f"| **{self._fmt(totals.get('option_b', 0), 1)}** "
                f"| **{self._fmt(totals.get('option_c', 0), 1)}** "
                f"| **{self._fmt(totals.get('option_d', 0), 1)}** |"
            )
        return "\n".join(lines)

    def _md_cost_effectiveness(self, data: Dict[str, Any]) -> str:
        """Render cost-effectiveness section."""
        costs = data.get("cost_effectiveness", [])
        if not costs:
            return "## 8. Cost-Effectiveness\n\n_No cost-effectiveness data available._"
        lines = [
            "## 8. Cost-Effectiveness\n",
            "| Option | M&V Cost | Savings Value | M&V/Savings (%) | NPV of M&V |",
            "|--------|--------:|-------------:|----------------:|----------:|",
        ]
        for c in costs:
            lines.append(
                f"| {c.get('option', '-')} "
                f"| {self._format_currency(c.get('mv_cost', 0))} "
                f"| {self._format_currency(c.get('savings_value', 0))} "
                f"| {self._fmt(c.get('mv_savings_ratio_pct', 0))}% "
                f"| {self._format_currency(c.get('npv_mv', 0))} |"
            )
        return "\n".join(lines)

    def _md_accuracy_tradeoffs(self, data: Dict[str, Any]) -> str:
        """Render accuracy trade-offs section."""
        tradeoffs = data.get("accuracy_tradeoffs", [])
        if not tradeoffs:
            return "## 9. Accuracy Trade-offs\n\n_No accuracy trade-off data available._"
        lines = [
            "## 9. Accuracy Trade-offs\n",
            "| Option | Accuracy (%) | Cost | Complexity | Risk | Overall |",
            "|--------|----------:|------|-----------|------|---------|",
        ]
        for t in tradeoffs:
            lines.append(
                f"| {t.get('option', '-')} "
                f"| {self._fmt(t.get('accuracy_pct', 0))}% "
                f"| {t.get('cost_level', '-')} "
                f"| {t.get('complexity', '-')} "
                f"| {t.get('risk_level', '-')} "
                f"| {t.get('overall_rating', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        """Render recommendation section."""
        rec = data.get("recommendation", {})
        if not rec:
            return "## 10. Recommendation\n\n_No recommendation data available._"
        return (
            "## 10. Recommendation\n\n"
            f"**Recommended Option:** {rec.get('option', '-')}  \n"
            f"**Option Name:** {rec.get('option_name', '-')}  \n"
            f"**Score:** {self._fmt(rec.get('score', 0), 1)}/10  \n"
            f"**Rationale:** {rec.get('rationale', '-')}  \n"
            f"**Key Considerations:** {rec.get('key_considerations', '-')}  \n"
            f"**Alternative:** {rec.get('alternative_option', '-')} (if constraints change)"
        )

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
            f'<h1>IPMVP Option Comparison Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'ECM: {data.get("ecm_name", "-")} | '
            f'Recommended: {data.get("recommended_option", "-")}</p>'
        )

    def _html_comparison_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML comparison overview cards."""
        o = data.get("comparison_overview", {})
        return (
            '<h2>1. Comparison Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">ECM Type</span>'
            f'<span class="value">{o.get("ecm_type", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Est. Savings</span>'
            f'<span class="value">{self._fmt(o.get("estimated_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Savings %</span>'
            f'<span class="value">{self._fmt(o.get("savings_fraction_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">M&amp;V Budget</span>'
            f'<span class="value">{self._format_currency(o.get("mv_budget", 0))}</span></div>\n'
            '</div>'
        )

    def _html_ecm_characteristics(self, data: Dict[str, Any]) -> str:
        """Render HTML ECM characteristics."""
        ecm = data.get("ecm_characteristics", {})
        return (
            '<h2>2. ECM Characteristics</h2>\n'
            '<table>\n'
            f'<tr><th>Attribute</th><th>Value</th></tr>\n'
            f'<tr><td>Description</td><td>{ecm.get("description", "-")}</td></tr>\n'
            f'<tr><td>Technology</td><td>{ecm.get("technology", "-")}</td></tr>\n'
            f'<tr><td>End Use</td><td>{ecm.get("end_use", "-")}</td></tr>\n'
            f'<tr><td>Expected Life</td><td>{ecm.get("expected_life_years", "-")} years</td></tr>\n'
            f'<tr><td>Capital Cost</td><td>{self._format_currency(ecm.get("capital_cost", 0))}</td></tr>\n'
            '</table>'
        )

    def _html_option_a(self, data: Dict[str, Any]) -> str:
        """Render HTML Option A table."""
        opt = data.get("option_a", {})
        return self._html_option_table("3. Option A - Key Parameter", opt)

    def _html_option_b(self, data: Dict[str, Any]) -> str:
        """Render HTML Option B table."""
        opt = data.get("option_b", {})
        return self._html_option_table("4. Option B - All Parameters", opt)

    def _html_option_c(self, data: Dict[str, Any]) -> str:
        """Render HTML Option C table."""
        opt = data.get("option_c", {})
        return self._html_option_table("5. Option C - Whole Facility", opt)

    def _html_option_d(self, data: Dict[str, Any]) -> str:
        """Render HTML Option D table."""
        opt = data.get("option_d", {})
        return self._html_option_table("6. Option D - Calibrated Simulation", opt)

    def _html_option_table(self, title: str, opt: Dict[str, Any]) -> str:
        """Render a generic option assessment HTML table."""
        score = opt.get("suitability_score", 0)
        cls = "severity-low" if score >= 7 else ("severity-medium" if score >= 4 else "severity-high")
        return (
            f'<h2>{title}</h2>\n'
            '<table>\n'
            f'<tr><th>Criterion</th><th>Assessment</th></tr>\n'
            f'<tr><td>Applicability</td><td>{opt.get("applicability", "-")}</td></tr>\n'
            f'<tr><td>Expected Accuracy</td><td>{self._fmt(opt.get("expected_accuracy_pct", 0))}%</td></tr>\n'
            f'<tr><td>M&amp;V Cost</td><td>{self._format_currency(opt.get("mv_cost", 0))}</td></tr>\n'
            f'<tr><td>Pros</td><td>{opt.get("pros", "-")}</td></tr>\n'
            f'<tr><td>Cons</td><td>{opt.get("cons", "-")}</td></tr>\n'
            f'<tr><td>Suitability</td><td class="{cls}">{self._fmt(score, 1)}/10</td></tr>\n'
            '</table>'
        )

    def _html_suitability_scoring(self, data: Dict[str, Any]) -> str:
        """Render HTML suitability scoring table."""
        scores = data.get("suitability_scoring", [])
        rows = ""
        for s in scores:
            rows += (
                f'<tr><td>{s.get("criterion", "-")}</td>'
                f'<td>{self._fmt(s.get("weight", 0), 1)}</td>'
                f'<td>{self._fmt(s.get("option_a", 0), 1)}</td>'
                f'<td>{self._fmt(s.get("option_b", 0), 1)}</td>'
                f'<td>{self._fmt(s.get("option_c", 0), 1)}</td>'
                f'<td>{self._fmt(s.get("option_d", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>7. Suitability Scoring</h2>\n'
            '<table>\n<tr><th>Criterion</th><th>Weight</th><th>Opt A</th>'
            f'<th>Opt B</th><th>Opt C</th><th>Opt D</th></tr>\n{rows}</table>'
        )

    def _html_cost_effectiveness(self, data: Dict[str, Any]) -> str:
        """Render HTML cost-effectiveness table."""
        costs = data.get("cost_effectiveness", [])
        rows = ""
        for c in costs:
            rows += (
                f'<tr><td>{c.get("option", "-")}</td>'
                f'<td>{self._format_currency(c.get("mv_cost", 0))}</td>'
                f'<td>{self._format_currency(c.get("savings_value", 0))}</td>'
                f'<td>{self._fmt(c.get("mv_savings_ratio_pct", 0))}%</td>'
                f'<td>{self._format_currency(c.get("npv_mv", 0))}</td></tr>\n'
            )
        return (
            '<h2>8. Cost-Effectiveness</h2>\n'
            '<table>\n<tr><th>Option</th><th>M&amp;V Cost</th><th>Savings Value</th>'
            f'<th>M&amp;V/Savings</th><th>NPV</th></tr>\n{rows}</table>'
        )

    def _html_accuracy_tradeoffs(self, data: Dict[str, Any]) -> str:
        """Render HTML accuracy trade-offs table."""
        tradeoffs = data.get("accuracy_tradeoffs", [])
        rows = ""
        for t in tradeoffs:
            rows += (
                f'<tr><td>{t.get("option", "-")}</td>'
                f'<td>{self._fmt(t.get("accuracy_pct", 0))}%</td>'
                f'<td>{t.get("cost_level", "-")}</td>'
                f'<td>{t.get("complexity", "-")}</td>'
                f'<td>{t.get("risk_level", "-")}</td>'
                f'<td>{t.get("overall_rating", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Accuracy Trade-offs</h2>\n'
            '<table>\n<tr><th>Option</th><th>Accuracy</th><th>Cost</th>'
            f'<th>Complexity</th><th>Risk</th><th>Overall</th></tr>\n{rows}</table>'
        )

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendation."""
        rec = data.get("recommendation", {})
        return (
            '<h2>10. Recommendation</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Recommended</span>'
            f'<span class="value">{rec.get("option", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Score</span>'
            f'<span class="value">{self._fmt(rec.get("score", 0), 1)}/10</span></div>\n'
            f'  <div class="card"><span class="label">Alternative</span>'
            f'<span class="value">{rec.get("alternative_option", "-")}</span></div>\n'
            '</div>\n'
            f'<p><strong>Rationale:</strong> {rec.get("rationale", "-")}</p>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON comparison overview."""
        o = data.get("comparison_overview", {})
        return {
            "ecm_type": o.get("ecm_type", ""),
            "estimated_savings_mwh": o.get("estimated_savings_mwh", 0),
            "savings_fraction_pct": o.get("savings_fraction_pct", 0),
            "isolation_possible": o.get("isolation_possible", False),
            "mv_budget": o.get("mv_budget", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        scores = data.get("suitability_scoring", [])
        costs = data.get("cost_effectiveness", [])
        return {
            "suitability_radar": {
                "type": "radar",
                "labels": [s.get("criterion", "") for s in scores],
                "series": {
                    "option_a": [s.get("option_a", 0) for s in scores],
                    "option_b": [s.get("option_b", 0) for s in scores],
                    "option_c": [s.get("option_c", 0) for s in scores],
                    "option_d": [s.get("option_d", 0) for s in scores],
                },
            },
            "cost_comparison": {
                "type": "bar",
                "labels": [c.get("option", "") for c in costs],
                "values": [c.get("mv_cost", 0) for c in costs],
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
