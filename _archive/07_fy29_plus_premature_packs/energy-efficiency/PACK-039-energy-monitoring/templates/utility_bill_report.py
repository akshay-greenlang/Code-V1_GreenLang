# -*- coding: utf-8 -*-
"""
UtilityBillReportTemplate - Utility bill validation for PACK-039.

Generates comprehensive utility bill validation reports showing
meter-to-bill reconciliation, rate schedule analysis, billing error
detection, historical bill comparison, and charge component breakdown
with variance tracking.

Sections:
    1. Bill Summary
    2. Meter-to-Bill Reconciliation
    3. Rate Analysis
    4. Billing Errors Detected
    5. Historical Comparison
    6. Charge Components
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy procurement and monitoring)
    - NAESB Standards (Utility billing and metering)
    - EN 17267 (Energy measurement and monitoring plan)

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class UtilityBillReportTemplate:
    """
    Utility bill validation report template.

    Renders utility bill validation reports showing meter-to-bill
    reconciliation, rate schedule analysis, billing error detection,
    historical bill comparison, and charge component breakdowns across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UtilityBillReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render utility bill report as Markdown.

        Args:
            data: Utility bill engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_bill_summary(data),
            self._md_meter_to_bill(data),
            self._md_rate_analysis(data),
            self._md_billing_errors(data),
            self._md_historical_comparison(data),
            self._md_charge_components(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render utility bill report as self-contained HTML.

        Args:
            data: Utility bill engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_bill_summary(data),
            self._html_meter_to_bill(data),
            self._html_rate_analysis(data),
            self._html_billing_errors(data),
            self._html_historical_comparison(data),
            self._html_charge_components(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Utility Bill Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render utility bill report as structured JSON.

        Args:
            data: Utility bill engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "utility_bill_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "bill_summary": self._json_bill_summary(data),
            "meter_to_bill": data.get("meter_to_bill", []),
            "rate_analysis": data.get("rate_analysis", {}),
            "billing_errors": data.get("billing_errors", []),
            "historical_comparison": data.get("historical_comparison", []),
            "charge_components": data.get("charge_components", []),
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
            f"# Utility Bill Validation Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Billing Period:** {data.get('billing_period', '')}  \n"
            f"**Utility Provider:** {data.get('utility_provider', '')}  \n"
            f"**Bill Amount:** {self._format_currency(data.get('bill_amount', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 UtilityBillReportTemplate v39.0.0\n\n---"
        )

    def _md_bill_summary(self, data: Dict[str, Any]) -> str:
        """Render bill summary section."""
        summary = data.get("bill_summary", {})
        return (
            "## 1. Bill Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Bill Amount | {self._format_currency(data.get('bill_amount', 0))} |\n"
            f"| Billed Consumption | {self._format_energy(summary.get('billed_consumption_mwh', 0))} |\n"
            f"| Metered Consumption | {self._format_energy(summary.get('metered_consumption_mwh', 0))} |\n"
            f"| Consumption Variance | {self._fmt(summary.get('consumption_variance_pct', 0))}% |\n"
            f"| Billed Demand | {self._format_power(summary.get('billed_demand_kw', 0))} |\n"
            f"| Metered Demand | {self._format_power(summary.get('metered_demand_kw', 0))} |\n"
            f"| Blended Rate | {self._format_currency(summary.get('blended_rate', 0))}/MWh |\n"
            f"| Errors Detected | {self._fmt(summary.get('errors_detected', 0), 0)} |"
        )

    def _md_meter_to_bill(self, data: Dict[str, Any]) -> str:
        """Render meter-to-bill reconciliation section."""
        meters = data.get("meter_to_bill", [])
        if not meters:
            return "## 2. Meter-to-Bill Reconciliation\n\n_No reconciliation data available._"
        lines = [
            "## 2. Meter-to-Bill Reconciliation\n",
            "| Meter ID | Metered (MWh) | Billed (MWh) | Variance (MWh) | Var % | Status |",
            "|----------|-------------:|------------:|--------------:|------:|--------|",
        ]
        for m in meters:
            metered = m.get("metered_mwh", 0)
            billed = m.get("billed_mwh", 0)
            variance = billed - metered
            var_pct = ((variance / metered) * 100) if metered != 0 else 0
            status = "OK" if abs(var_pct) < 2 else "Review"
            lines.append(
                f"| {m.get('meter_id', '-')} "
                f"| {self._fmt(metered, 2)} "
                f"| {self._fmt(billed, 2)} "
                f"| {self._fmt(variance, 2)} "
                f"| {self._fmt(var_pct)}% "
                f"| {m.get('status', status)} |"
            )
        return "\n".join(lines)

    def _md_rate_analysis(self, data: Dict[str, Any]) -> str:
        """Render rate analysis section."""
        rate = data.get("rate_analysis", {})
        if not rate:
            return "## 3. Rate Analysis\n\n_No rate analysis data available._"
        schedules = rate.get("schedules", [])
        lines = [
            "## 3. Rate Analysis\n",
            f"**Tariff:** {rate.get('tariff_name', '-')}  \n"
            f"**Rate Class:** {rate.get('rate_class', '-')}  \n"
            f"**Effective Date:** {rate.get('effective_date', '-')}\n",
        ]
        if schedules:
            lines.append("### Rate Schedules\n")
            lines.append("| Period | Rate (EUR/kWh) | Consumption (kWh) | Cost | % of Total |")
            lines.append("|--------|-------------:|------------------:|-----:|----------:|")
            total_cost = rate.get("total_energy_cost", 1)
            for s in schedules:
                cost = s.get("cost", 0)
                lines.append(
                    f"| {s.get('period', '-')} "
                    f"| {self._fmt(s.get('rate', 0), 4)} "
                    f"| {self._fmt(s.get('consumption_kwh', 0), 0)} "
                    f"| {self._format_currency(cost)} "
                    f"| {self._pct(cost, total_cost)} |"
                )
        return "\n".join(lines)

    def _md_billing_errors(self, data: Dict[str, Any]) -> str:
        """Render billing errors detected section."""
        errors = data.get("billing_errors", [])
        if not errors:
            return "## 4. Billing Errors Detected\n\n_No billing errors detected._"
        lines = [
            "## 4. Billing Errors Detected\n",
            "| # | Error Type | Description | Impact | Confidence | Status |",
            "|---|-----------|-------------|-------:|----------:|--------|",
        ]
        for i, err in enumerate(errors, 1):
            lines.append(
                f"| {i} | {err.get('error_type', '-')} "
                f"| {err.get('description', '-')} "
                f"| {self._format_currency(err.get('impact', 0))} "
                f"| {self._fmt(err.get('confidence', 0))}% "
                f"| {err.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_historical_comparison(self, data: Dict[str, Any]) -> str:
        """Render historical comparison section."""
        history = data.get("historical_comparison", [])
        if not history:
            return "## 5. Historical Comparison\n\n_No historical comparison data available._"
        lines = [
            "## 5. Historical Comparison\n",
            "| Period | Bill Amount | Consumption (MWh) | Blended Rate | vs Prev Period |",
            "|--------|----------:|------------------:|------------:|---------------:|",
        ]
        for h in history:
            lines.append(
                f"| {h.get('period', '-')} "
                f"| {self._format_currency(h.get('bill_amount', 0))} "
                f"| {self._fmt(h.get('consumption_mwh', 0), 1)} "
                f"| {self._format_currency(h.get('blended_rate', 0))}/MWh "
                f"| {self._fmt(h.get('vs_prev_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_charge_components(self, data: Dict[str, Any]) -> str:
        """Render charge components section."""
        components = data.get("charge_components", [])
        if not components:
            return "## 6. Charge Components\n\n_No charge component data available._"
        total = data.get("bill_amount", 1)
        lines = [
            "## 6. Charge Components\n",
            "| Component | Amount | % of Bill | vs Previous | Notes |",
            "|-----------|-------:|----------:|----------:|-------|",
        ]
        for c in components:
            amount = c.get("amount", 0)
            lines.append(
                f"| {c.get('component', '-')} "
                f"| {self._format_currency(amount)} "
                f"| {self._pct(amount, total)} "
                f"| {self._fmt(c.get('vs_prev_pct', 0))}% "
                f"| {c.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Dispute detected billing errors with utility provider",
                "Review rate schedule for optimization opportunities",
                "Install check meters for high-value supply points",
                "Automate bill validation to catch errors within dispute window",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-039 Energy Monitoring Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Utility Bill Validation Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Provider: {data.get("utility_provider", "-")} | '
            f'Period: {data.get("billing_period", "-")}</p>'
        )

    def _html_bill_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML bill summary cards."""
        s = data.get("bill_summary", {})
        errors = s.get("errors_detected", 0)
        err_cls = "severity-high" if errors > 0 else ""
        return (
            '<h2>Bill Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Bill Amount</span>'
            f'<span class="value">{self._format_currency(data.get("bill_amount", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Billed Consumption</span>'
            f'<span class="value">{self._fmt(s.get("billed_consumption_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Metered Consumption</span>'
            f'<span class="value">{self._fmt(s.get("metered_consumption_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Blended Rate</span>'
            f'<span class="value">{self._format_currency(s.get("blended_rate", 0))}/MWh</span></div>\n'
            f'  <div class="card"><span class="label">Errors</span>'
            f'<span class="value {err_cls}">{errors}</span></div>\n'
            '</div>'
        )

    def _html_meter_to_bill(self, data: Dict[str, Any]) -> str:
        """Render HTML meter-to-bill reconciliation table."""
        meters = data.get("meter_to_bill", [])
        rows = ""
        for m in meters:
            metered = m.get("metered_mwh", 0)
            billed = m.get("billed_mwh", 0)
            variance = billed - metered
            var_pct = ((variance / metered) * 100) if metered != 0 else 0
            cls = "severity-high" if abs(var_pct) >= 2 else ""
            rows += (
                f'<tr><td>{m.get("meter_id", "-")}</td>'
                f'<td>{self._fmt(metered, 2)}</td>'
                f'<td>{self._fmt(billed, 2)}</td>'
                f'<td class="{cls}">{self._fmt(variance, 2)}</td>'
                f'<td class="{cls}">{self._fmt(var_pct)}%</td></tr>\n'
            )
        return (
            '<h2>Meter-to-Bill Reconciliation</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Metered (MWh)</th>'
            '<th>Billed (MWh)</th><th>Variance (MWh)</th>'
            f'<th>Var %</th></tr>\n{rows}</table>'
        )

    def _html_rate_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML rate analysis table."""
        rate = data.get("rate_analysis", {})
        schedules = rate.get("schedules", [])
        rows = ""
        for s in schedules:
            rows += (
                f'<tr><td>{s.get("period", "-")}</td>'
                f'<td>{self._fmt(s.get("rate", 0), 4)}</td>'
                f'<td>{self._fmt(s.get("consumption_kwh", 0), 0)}</td>'
                f'<td>{self._format_currency(s.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Rate Analysis</h2>\n'
            f'<p>Tariff: {rate.get("tariff_name", "-")} | '
            f'Rate Class: {rate.get("rate_class", "-")}</p>\n'
            '<table>\n<tr><th>Period</th><th>Rate (EUR/kWh)</th>'
            f'<th>Consumption (kWh)</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_billing_errors(self, data: Dict[str, Any]) -> str:
        """Render HTML billing errors table."""
        errors = data.get("billing_errors", [])
        rows = ""
        for i, err in enumerate(errors, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td class="severity-high">{err.get("error_type", "-")}</td>'
                f'<td>{err.get("description", "-")}</td>'
                f'<td>{self._format_currency(err.get("impact", 0))}</td>'
                f'<td>{self._fmt(err.get("confidence", 0))}%</td>'
                f'<td>{err.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Billing Errors Detected</h2>\n'
            '<table>\n<tr><th>#</th><th>Error Type</th><th>Description</th>'
            f'<th>Impact</th><th>Confidence</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_historical_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML historical comparison table."""
        history = data.get("historical_comparison", [])
        rows = ""
        for h in history:
            rows += (
                f'<tr><td>{h.get("period", "-")}</td>'
                f'<td>{self._format_currency(h.get("bill_amount", 0))}</td>'
                f'<td>{self._fmt(h.get("consumption_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(h.get("blended_rate", 0))}</td>'
                f'<td>{self._fmt(h.get("vs_prev_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Historical Comparison</h2>\n'
            '<table>\n<tr><th>Period</th><th>Bill Amount</th>'
            '<th>Consumption (MWh)</th><th>Blended Rate</th>'
            f'<th>vs Previous</th></tr>\n{rows}</table>'
        )

    def _html_charge_components(self, data: Dict[str, Any]) -> str:
        """Render HTML charge components table."""
        components = data.get("charge_components", [])
        rows = ""
        for c in components:
            rows += (
                f'<tr><td>{c.get("component", "-")}</td>'
                f'<td>{self._format_currency(c.get("amount", 0))}</td>'
                f'<td>{self._fmt(c.get("vs_prev_pct", 0))}%</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )
        return (
            '<h2>Charge Components</h2>\n'
            '<table>\n<tr><th>Component</th><th>Amount</th>'
            f'<th>vs Previous</th><th>Notes</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Dispute detected billing errors with utility provider",
            "Review rate schedule for optimization opportunities",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_bill_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON bill summary."""
        s = data.get("bill_summary", {})
        return {
            "bill_amount": data.get("bill_amount", 0),
            "billed_consumption_mwh": s.get("billed_consumption_mwh", 0),
            "metered_consumption_mwh": s.get("metered_consumption_mwh", 0),
            "consumption_variance_pct": s.get("consumption_variance_pct", 0),
            "billed_demand_kw": s.get("billed_demand_kw", 0),
            "metered_demand_kw": s.get("metered_demand_kw", 0),
            "blended_rate": s.get("blended_rate", 0),
            "errors_detected": s.get("errors_detected", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        history = data.get("historical_comparison", [])
        components = data.get("charge_components", [])
        meters = data.get("meter_to_bill", [])
        return {
            "bill_history": {
                "type": "line",
                "labels": [h.get("period", "") for h in history],
                "values": [h.get("bill_amount", 0) for h in history],
            },
            "charge_breakdown": {
                "type": "pie",
                "labels": [c.get("component", "") for c in components],
                "values": [c.get("amount", 0) for c in components],
            },
            "meter_variance": {
                "type": "bar",
                "labels": [m.get("meter_id", "") for m in meters],
                "series": {
                    "metered": [m.get("metered_mwh", 0) for m in meters],
                    "billed": [m.get("billed_mwh", 0) for m in meters],
                },
            },
            "rate_trend": {
                "type": "line",
                "labels": [h.get("period", "") for h in history],
                "values": [h.get("blended_rate", 0) for h in history],
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
