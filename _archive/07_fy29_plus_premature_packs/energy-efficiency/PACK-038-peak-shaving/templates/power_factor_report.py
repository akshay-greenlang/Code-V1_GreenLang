# -*- coding: utf-8 -*-
"""
PowerFactorReportTemplate - Power factor analysis for PACK-038.

Generates comprehensive power factor analysis reports showing PF
profile across billing periods, reactive demand quantification,
correction equipment sizing recommendations, penalty savings
calculations, and harmonic distortion assessment with THD metrics.

Sections:
    1. PF Summary
    2. PF Profile Analysis
    3. Reactive Demand
    4. Correction Sizing
    5. Penalty Savings
    6. Harmonic Assessment
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IEEE Std 1459 (power measurement definitions)
    - IEC 61000-3-2 (harmonic current emissions)
    - EN 50160 (voltage characteristics of supply)
    - IEEE Std 519 (harmonic control in power systems)

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


class PowerFactorReportTemplate:
    """
    Power factor analysis report template.

    Renders power factor analysis reports showing PF profiles,
    reactive demand, correction equipment sizing, penalty savings,
    and harmonic assessment across markdown, HTML, and JSON formats.
    All outputs include SHA-256 provenance hashing for audit trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PowerFactorReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render power factor report as Markdown.

        Args:
            data: Power factor engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_pf_summary(data),
            self._md_pf_profile(data),
            self._md_reactive_demand(data),
            self._md_correction_sizing(data),
            self._md_penalty_savings(data),
            self._md_harmonic_assessment(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render power factor report as self-contained HTML.

        Args:
            data: Power factor engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_pf_summary(data),
            self._html_pf_profile(data),
            self._html_reactive_demand(data),
            self._html_correction_sizing(data),
            self._html_penalty_savings(data),
            self._html_harmonic_assessment(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Power Factor Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render power factor report as structured JSON.

        Args:
            data: Power factor engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "power_factor_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "pf_summary": self._json_pf_summary(data),
            "pf_profile": data.get("pf_profile", []),
            "reactive_demand": self._json_reactive_demand(data),
            "correction_sizing": data.get("correction_sizing", []),
            "penalty_savings": self._json_penalty_savings(data),
            "harmonic_assessment": data.get("harmonic_assessment", []),
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
        """Render markdown header with facility metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Power Factor Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Average PF:** {self._fmt(data.get('average_pf', 0), 3)}  \n"
            f"**Utility PF Threshold:** {self._fmt(data.get('pf_threshold', 0), 2)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 PowerFactorReportTemplate v38.0.0\n\n---"
        )

    def _md_pf_summary(self, data: Dict[str, Any]) -> str:
        """Render PF summary section."""
        summary = data.get("pf_summary", {})
        return (
            "## 1. PF Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Average Power Factor | {self._fmt(summary.get('average_pf', 0), 3)} |\n"
            f"| Minimum Power Factor | {self._fmt(summary.get('minimum_pf', 0), 3)} |\n"
            f"| Maximum Power Factor | {self._fmt(summary.get('maximum_pf', 0), 3)} |\n"
            f"| Months Below Threshold | {summary.get('months_below_threshold', 0)} |\n"
            f"| Total Reactive Demand (kVAR) | {self._fmt(summary.get('total_kvar', 0), 1)} |\n"
            f"| Annual PF Penalties | {self._format_currency(summary.get('annual_penalties', 0))} |\n"
            f"| Correction CAPEX | {self._format_currency(summary.get('correction_capex', 0))} |\n"
            f"| Target PF After Correction | {self._fmt(summary.get('target_pf', 0), 3)} |"
        )

    def _md_pf_profile(self, data: Dict[str, Any]) -> str:
        """Render PF profile analysis section."""
        profile = data.get("pf_profile", [])
        if not profile:
            return "## 2. PF Profile Analysis\n\n_No PF profile data available._"
        lines = [
            "## 2. PF Profile Analysis\n",
            "| Month | PF | kW | kVAR | kVA | Below Threshold |",
            "|-------|---:|---:|-----:|----:|:---------------:|",
        ]
        for p in profile:
            threshold = data.get("pf_threshold", 0.9)
            below = "Yes" if p.get("pf", 1.0) < threshold else "No"
            lines.append(
                f"| {p.get('month', '-')} "
                f"| {self._fmt(p.get('pf', 0), 3)} "
                f"| {self._fmt(p.get('kw', 0), 1)} "
                f"| {self._fmt(p.get('kvar', 0), 1)} "
                f"| {self._fmt(p.get('kva', 0), 1)} "
                f"| {below} |"
            )
        return "\n".join(lines)

    def _md_reactive_demand(self, data: Dict[str, Any]) -> str:
        """Render reactive demand section."""
        reactive = data.get("reactive_demand", {})
        if not reactive:
            return "## 3. Reactive Demand\n\n_No reactive demand data available._"
        return (
            "## 3. Reactive Demand\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Reactive Demand | {self._fmt(reactive.get('total_kvar', 0), 1)} kVAR |\n"
            f"| Peak Reactive Demand | {self._fmt(reactive.get('peak_kvar', 0), 1)} kVAR |\n"
            f"| Motor Contribution | {self._fmt(reactive.get('motor_kvar', 0), 1)} kVAR ({self._fmt(reactive.get('motor_pct', 0))}%) |\n"
            f"| Transformer Contribution | {self._fmt(reactive.get('transformer_kvar', 0), 1)} kVAR ({self._fmt(reactive.get('transformer_pct', 0))}%) |\n"
            f"| Lighting Contribution | {self._fmt(reactive.get('lighting_kvar', 0), 1)} kVAR ({self._fmt(reactive.get('lighting_pct', 0))}%) |\n"
            f"| Other Contribution | {self._fmt(reactive.get('other_kvar', 0), 1)} kVAR ({self._fmt(reactive.get('other_pct', 0))}%) |\n"
            f"| Required Correction (kVAR) | {self._fmt(reactive.get('required_correction_kvar', 0), 1)} kVAR |"
        )

    def _md_correction_sizing(self, data: Dict[str, Any]) -> str:
        """Render correction sizing section."""
        options = data.get("correction_sizing", [])
        if not options:
            return "## 4. Correction Sizing\n\n_No correction sizing data available._"
        lines = [
            "## 4. Correction Sizing\n",
            "| Option | Type | kVAR | Target PF | CAPEX | Annual Savings | Payback |",
            "|--------|------|-----:|--------:|------:|-------------:|--------:|",
        ]
        for opt in options:
            lines.append(
                f"| {opt.get('option', '-')} "
                f"| {opt.get('type', '-')} "
                f"| {self._fmt(opt.get('kvar', 0), 0)} "
                f"| {self._fmt(opt.get('target_pf', 0), 3)} "
                f"| {self._format_currency(opt.get('capex', 0))} "
                f"| {self._format_currency(opt.get('annual_savings', 0))} "
                f"| {self._fmt(opt.get('payback_months', 0), 0)} mo |"
            )
        return "\n".join(lines)

    def _md_penalty_savings(self, data: Dict[str, Any]) -> str:
        """Render penalty savings section."""
        savings = data.get("penalty_savings", {})
        if not savings:
            return "## 5. Penalty Savings\n\n_No penalty savings data available._"
        return (
            "## 5. Penalty Savings\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Current Annual Penalties | {self._format_currency(savings.get('current_annual_penalties', 0))} |\n"
            f"| Projected Penalties (Post-Fix) | {self._format_currency(savings.get('projected_penalties', 0))} |\n"
            f"| Annual Penalty Savings | {self._format_currency(savings.get('annual_penalty_savings', 0))} |\n"
            f"| Demand Charge Reduction | {self._format_currency(savings.get('demand_charge_reduction', 0))} |\n"
            f"| Line Loss Savings | {self._format_currency(savings.get('line_loss_savings', 0))} |\n"
            f"| Total Annual Savings | {self._format_currency(savings.get('total_annual_savings', 0))} |\n"
            f"| 5-Year NPV | {self._format_currency(savings.get('five_year_npv', 0))} |"
        )

    def _md_harmonic_assessment(self, data: Dict[str, Any]) -> str:
        """Render harmonic assessment section."""
        harmonics = data.get("harmonic_assessment", [])
        if not harmonics:
            return "## 6. Harmonic Assessment\n\n_No harmonic data available._"
        lines = [
            "## 6. Harmonic Assessment\n",
            "| Harmonic | Frequency (Hz) | THD (%) | Limit (%) | Status | Source |",
            "|----------|-------------:|-------:|--------:|--------|--------|",
        ]
        for h in harmonics:
            status = "PASS" if h.get("thd_pct", 0) <= h.get("limit_pct", 100) else "FAIL"
            lines.append(
                f"| {h.get('harmonic', '-')} "
                f"| {self._fmt(h.get('frequency_hz', 0), 0)} "
                f"| {self._fmt(h.get('thd_pct', 0), 2)} "
                f"| {self._fmt(h.get('limit_pct', 0), 2)} "
                f"| {status} "
                f"| {h.get('source', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Install automatic power factor correction capacitor banks",
                "Add harmonic filters for VFD-heavy load groups",
                "Monitor PF continuously with smart metering integration",
                "Review tariff structure for optimal PF threshold targets",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
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
            f'<h1>Power Factor Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Average PF: {self._fmt(data.get("average_pf", 0), 3)} | '
            f'Threshold: {self._fmt(data.get("pf_threshold", 0), 2)}</p>'
        )

    def _html_pf_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML PF summary cards."""
        s = data.get("pf_summary", {})
        return (
            '<h2>PF Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Average PF</span>'
            f'<span class="value">{self._fmt(s.get("average_pf", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Min PF</span>'
            f'<span class="value">{self._fmt(s.get("minimum_pf", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Months Below</span>'
            f'<span class="value">{s.get("months_below_threshold", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Penalties</span>'
            f'<span class="value">{self._format_currency(s.get("annual_penalties", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Correction CAPEX</span>'
            f'<span class="value">{self._format_currency(s.get("correction_capex", 0))}</span></div>\n'
            '</div>'
        )

    def _html_pf_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML PF profile table."""
        profile = data.get("pf_profile", [])
        threshold = data.get("pf_threshold", 0.9)
        rows = ""
        for p in profile:
            pf_val = p.get("pf", 1.0)
            color = "#dc3545" if pf_val < threshold else "#198754"
            rows += (
                f'<tr><td>{p.get("month", "-")}</td>'
                f'<td style="color:{color};font-weight:700">{self._fmt(pf_val, 3)}</td>'
                f'<td>{self._fmt(p.get("kw", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("kvar", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("kva", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>PF Profile Analysis</h2>\n'
            '<table>\n<tr><th>Month</th><th>PF</th><th>kW</th>'
            f'<th>kVAR</th><th>kVA</th></tr>\n{rows}</table>'
        )

    def _html_reactive_demand(self, data: Dict[str, Any]) -> str:
        """Render HTML reactive demand breakdown."""
        r = data.get("reactive_demand", {})
        return (
            '<h2>Reactive Demand</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total kVAR</span>'
            f'<span class="value">{self._fmt(r.get("total_kvar", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Peak kVAR</span>'
            f'<span class="value">{self._fmt(r.get("peak_kvar", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Required Correction</span>'
            f'<span class="value">{self._fmt(r.get("required_correction_kvar", 0), 0)} kVAR</span></div>\n'
            '</div>'
        )

    def _html_correction_sizing(self, data: Dict[str, Any]) -> str:
        """Render HTML correction sizing table."""
        options = data.get("correction_sizing", [])
        rows = ""
        for opt in options:
            rows += (
                f'<tr><td>{opt.get("option", "-")}</td>'
                f'<td>{opt.get("type", "-")}</td>'
                f'<td>{self._fmt(opt.get("kvar", 0), 0)}</td>'
                f'<td>{self._fmt(opt.get("target_pf", 0), 3)}</td>'
                f'<td>{self._format_currency(opt.get("capex", 0))}</td>'
                f'<td>{self._format_currency(opt.get("annual_savings", 0))}</td>'
                f'<td>{self._fmt(opt.get("payback_months", 0), 0)} mo</td></tr>\n'
            )
        return (
            '<h2>Correction Sizing</h2>\n'
            '<table>\n<tr><th>Option</th><th>Type</th><th>kVAR</th>'
            f'<th>Target PF</th><th>CAPEX</th><th>Savings</th><th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_penalty_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML penalty savings summary."""
        s = data.get("penalty_savings", {})
        return (
            '<h2>Penalty Savings</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Current Penalties</span>'
            f'<span class="value">{self._format_currency(s.get("current_annual_penalties", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Penalty Savings</span>'
            f'<span class="value">{self._format_currency(s.get("annual_penalty_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Total Savings</span>'
            f'<span class="value">{self._format_currency(s.get("total_annual_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">5-Year NPV</span>'
            f'<span class="value">{self._format_currency(s.get("five_year_npv", 0))}</span></div>\n'
            '</div>'
        )

    def _html_harmonic_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML harmonic assessment table."""
        harmonics = data.get("harmonic_assessment", [])
        rows = ""
        for h in harmonics:
            status = "PASS" if h.get("thd_pct", 0) <= h.get("limit_pct", 100) else "FAIL"
            color = "#198754" if status == "PASS" else "#dc3545"
            rows += (
                f'<tr><td>{h.get("harmonic", "-")}</td>'
                f'<td>{self._fmt(h.get("frequency_hz", 0), 0)}</td>'
                f'<td>{self._fmt(h.get("thd_pct", 0), 2)}%</td>'
                f'<td>{self._fmt(h.get("limit_pct", 0), 2)}%</td>'
                f'<td style="color:{color};font-weight:700">{status}</td>'
                f'<td>{h.get("source", "-")}</td></tr>\n'
            )
        return (
            '<h2>Harmonic Assessment</h2>\n'
            '<table>\n<tr><th>Harmonic</th><th>Freq (Hz)</th><th>THD</th>'
            f'<th>Limit</th><th>Status</th><th>Source</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Install automatic power factor correction capacitor banks",
            "Add harmonic filters for VFD-heavy load groups",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_pf_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON PF summary."""
        s = data.get("pf_summary", {})
        return {
            "average_pf": s.get("average_pf", 0),
            "minimum_pf": s.get("minimum_pf", 0),
            "maximum_pf": s.get("maximum_pf", 0),
            "months_below_threshold": s.get("months_below_threshold", 0),
            "total_kvar": s.get("total_kvar", 0),
            "annual_penalties": s.get("annual_penalties", 0),
            "correction_capex": s.get("correction_capex", 0),
            "target_pf": s.get("target_pf", 0),
        }

    def _json_reactive_demand(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON reactive demand."""
        r = data.get("reactive_demand", {})
        return {
            "total_kvar": r.get("total_kvar", 0),
            "peak_kvar": r.get("peak_kvar", 0),
            "motor_kvar": r.get("motor_kvar", 0),
            "transformer_kvar": r.get("transformer_kvar", 0),
            "lighting_kvar": r.get("lighting_kvar", 0),
            "other_kvar": r.get("other_kvar", 0),
            "required_correction_kvar": r.get("required_correction_kvar", 0),
        }

    def _json_penalty_savings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON penalty savings."""
        s = data.get("penalty_savings", {})
        return {
            "current_annual_penalties": s.get("current_annual_penalties", 0),
            "projected_penalties": s.get("projected_penalties", 0),
            "annual_penalty_savings": s.get("annual_penalty_savings", 0),
            "demand_charge_reduction": s.get("demand_charge_reduction", 0),
            "line_loss_savings": s.get("line_loss_savings", 0),
            "total_annual_savings": s.get("total_annual_savings", 0),
            "five_year_npv": s.get("five_year_npv", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        profile = data.get("pf_profile", [])
        reactive = data.get("reactive_demand", {})
        harmonics = data.get("harmonic_assessment", [])
        return {
            "pf_trend_line": {
                "type": "line",
                "labels": [p.get("month", "") for p in profile],
                "values": [p.get("pf", 0) for p in profile],
                "threshold": data.get("pf_threshold", 0.9),
            },
            "reactive_pie": {
                "type": "pie",
                "labels": ["Motors", "Transformers", "Lighting", "Other"],
                "values": [
                    reactive.get("motor_kvar", 0),
                    reactive.get("transformer_kvar", 0),
                    reactive.get("lighting_kvar", 0),
                    reactive.get("other_kvar", 0),
                ],
            },
            "harmonic_spectrum": {
                "type": "bar",
                "labels": [h.get("harmonic", "") for h in harmonics],
                "series": {
                    "thd": [h.get("thd_pct", 0) for h in harmonics],
                    "limit": [h.get("limit_pct", 0) for h in harmonics],
                },
            },
            "kvar_profile": {
                "type": "area",
                "labels": [p.get("month", "") for p in profile],
                "values": [p.get("kvar", 0) for p in profile],
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
