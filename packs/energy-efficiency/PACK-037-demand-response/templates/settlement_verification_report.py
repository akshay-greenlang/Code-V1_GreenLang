# -*- coding: utf-8 -*-
"""
SettlementVerificationReportTemplate - Settlement-grade DR documentation for PACK-037.

Generates settlement-grade verification reports for demand response program
administrators with detailed baseline calculations, interval meter data,
performance verification, adjustment documentation, and audit-ready
settlement records.

Sections:
    1. Settlement Summary
    2. Baseline Calculation Documentation
    3. Interval Meter Data
    4. Performance Verification
    5. Adjustment Documentation
    6. Settlement Calculations
    7. Certification & Attestation

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - NAESB WEQ Business Practice Standards
    - PJM Manual 11 (Energy & Ancillary Services)
    - NYISO Accounting & Billing Manual
    - ISO-NE Manual M-28 (Market Settlement)

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class SettlementVerificationReportTemplate:
    """
    Settlement-grade verification report template.

    Renders settlement documentation for DR program administrators with
    baseline calculations, meter data, performance verification, and
    audit-ready records across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SettlementVerificationReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render settlement verification report as Markdown.

        Args:
            data: Settlement verification engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_settlement_summary(data),
            self._md_baseline_documentation(data),
            self._md_interval_meter_data(data),
            self._md_performance_verification(data),
            self._md_adjustment_documentation(data),
            self._md_settlement_calculations(data),
            self._md_certification(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render settlement verification report as self-contained HTML.

        Args:
            data: Settlement verification engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_settlement_summary(data),
            self._html_baseline_documentation(data),
            self._html_interval_meter_data(data),
            self._html_performance_verification(data),
            self._html_adjustment_documentation(data),
            self._html_settlement_calculations(data),
            self._html_certification(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Settlement Verification Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render settlement verification report as structured JSON.

        Args:
            data: Settlement verification engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "settlement_verification_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "settlement_summary": self._json_settlement_summary(data),
            "baseline_documentation": data.get("baseline_documentation", {}),
            "interval_meter_data": data.get("interval_meter_data", {}),
            "performance_verification": data.get("performance_verification", {}),
            "adjustment_documentation": data.get("adjustment_documentation", []),
            "settlement_calculations": data.get("settlement_calculations", {}),
            "certification": data.get("certification", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Settlement Verification Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Account ID:** {data.get('account_id', '')}  \n"
            f"**Event ID:** {data.get('event_id', '')}  \n"
            f"**Program:** {data.get('program_name', '')}  \n"
            f"**Event Date:** {data.get('event_date', '')}  \n"
            f"**Meter ID:** {data.get('meter_id', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 SettlementVerificationReportTemplate v37.0.0\n\n---"
        )

    def _md_settlement_summary(self, data: Dict[str, Any]) -> str:
        """Render settlement summary section."""
        summary = data.get("settlement_summary", {})
        return (
            "## 1. Settlement Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Event Window | {summary.get('event_window', '-')} |\n"
            f"| Event Duration | {summary.get('duration_hours', 0)} hours |\n"
            f"| Baseline Methodology | {summary.get('baseline_methodology', '-')} |\n"
            f"| CBL Average (kW) | {self._format_power(summary.get('cbl_average_kw', 0))} |\n"
            f"| Actual Average (kW) | {self._format_power(summary.get('actual_average_kw', 0))} |\n"
            f"| Verified Curtailment (kW) | {self._format_power(summary.get('verified_curtailment_kw', 0))} |\n"
            f"| Committed Curtailment (kW) | {self._format_power(summary.get('committed_curtailment_kw', 0))} |\n"
            f"| Performance Factor | {self._fmt(summary.get('performance_factor', 0), 4)} |\n"
            f"| Settlement Amount | {self._format_currency(summary.get('settlement_amount', 0))} |\n"
            f"| Settlement Status | {summary.get('settlement_status', '-')} |"
        )

    def _md_baseline_documentation(self, data: Dict[str, Any]) -> str:
        """Render baseline calculation documentation section."""
        baseline = data.get("baseline_documentation", {})
        days = baseline.get("baseline_days", [])
        lines = [
            "## 2. Baseline Calculation Documentation\n",
            f"**Methodology:** {baseline.get('methodology', '-')}  ",
            f"**Selection Criteria:** {baseline.get('selection_criteria', '-')}  ",
            f"**Excluded Days:** {baseline.get('excluded_days_count', 0)}  ",
            f"**Adjustment Method:** {baseline.get('adjustment_method', 'None')}\n",
        ]
        if days:
            lines.extend([
                "### Baseline Reference Days\n",
                "| Date | Day Type | Peak kW | Included | Exclusion Reason |",
                "|------|----------|-------:|----------|-----------------|",
            ])
            for d in days:
                included = "Yes" if d.get("included", True) else "No"
                lines.append(
                    f"| {d.get('date', '-')} "
                    f"| {d.get('day_type', '-')} "
                    f"| {self._fmt(d.get('peak_kw', 0), 1)} "
                    f"| {included} "
                    f"| {d.get('exclusion_reason', '-')} |"
                )
        return "\n".join(lines)

    def _md_interval_meter_data(self, data: Dict[str, Any]) -> str:
        """Render interval meter data section."""
        meter = data.get("interval_meter_data", {})
        intervals = meter.get("intervals", [])
        if not intervals:
            return "## 3. Interval Meter Data\n\n_No interval data available._"
        lines = [
            "## 3. Interval Meter Data\n",
            f"**Meter ID:** {meter.get('meter_id', '-')}  ",
            f"**Interval Length:** {meter.get('interval_minutes', 5)} minutes  ",
            f"**Data Source:** {meter.get('data_source', '-')}  ",
            f"**Data Quality:** {meter.get('data_quality', '-')}\n",
            "| Interval | Baseline (kW) | Actual (kW) | Curtailment (kW) | Validated |",
            "|----------|-------------:|------------:|------------------:|----------|",
        ]
        for iv in intervals:
            baseline_kw = iv.get("baseline_kw", 0)
            actual_kw = iv.get("actual_kw", 0)
            curtailment = baseline_kw - actual_kw
            validated = "Yes" if iv.get("validated", True) else "No"
            lines.append(
                f"| {iv.get('interval', '-')} "
                f"| {self._fmt(baseline_kw, 1)} "
                f"| {self._fmt(actual_kw, 1)} "
                f"| {self._fmt(curtailment, 1)} "
                f"| {validated} |"
            )
        return "\n".join(lines)

    def _md_performance_verification(self, data: Dict[str, Any]) -> str:
        """Render performance verification section."""
        perf = data.get("performance_verification", {})
        return (
            "## 4. Performance Verification\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Committed Capacity (kW) | {self._format_power(perf.get('committed_capacity_kw', 0))} |\n"
            f"| Verified Curtailment (kW) | {self._format_power(perf.get('verified_curtailment_kw', 0))} |\n"
            f"| Performance Ratio | {self._fmt(perf.get('performance_ratio', 0), 4)} |\n"
            f"| Compliance Threshold | {self._fmt(perf.get('compliance_threshold', 0), 2)} |\n"
            f"| Compliance Status | {perf.get('compliance_status', '-')} |\n"
            f"| Min Interval Curtailment | {self._format_power(perf.get('min_interval_curtailment_kw', 0))} |\n"
            f"| Max Interval Curtailment | {self._format_power(perf.get('max_interval_curtailment_kw', 0))} |\n"
            f"| Std Deviation | {self._fmt(perf.get('std_deviation_kw', 0), 1)} kW |\n"
            f"| Verification Method | {perf.get('verification_method', '-')} |"
        )

    def _md_adjustment_documentation(self, data: Dict[str, Any]) -> str:
        """Render adjustment documentation section."""
        adjustments = data.get("adjustment_documentation", [])
        if not adjustments:
            return "## 5. Adjustment Documentation\n\n_No adjustments applied._"
        lines = [
            "## 5. Adjustment Documentation\n",
            "| # | Adjustment Type | Factor | Impact (kW) | Justification | Approved |",
            "|---|----------------|-------:|----------:|--------------|----------|",
        ]
        for i, adj in enumerate(adjustments, 1):
            approved = "Yes" if adj.get("approved", True) else "No"
            lines.append(
                f"| {i} | {adj.get('type', '-')} "
                f"| {self._fmt(adj.get('factor', 1.0), 4)} "
                f"| {self._fmt(adj.get('impact_kw', 0), 1)} "
                f"| {adj.get('justification', '-')} "
                f"| {approved} |"
            )
        return "\n".join(lines)

    def _md_settlement_calculations(self, data: Dict[str, Any]) -> str:
        """Render settlement calculations section."""
        calc = data.get("settlement_calculations", {})
        lines_items = calc.get("line_items", [])
        lines = [
            "## 6. Settlement Calculations\n",
            f"**Settlement Period:** {calc.get('settlement_period', '-')}  ",
            f"**Price Node:** {calc.get('price_node', '-')}  ",
            f"**Settlement Price:** {self._format_currency(calc.get('settlement_price_per_kwh', 0))}/kWh\n",
        ]
        if lines_items:
            lines.extend([
                "| Line Item | Quantity | Rate | Amount |",
                "|-----------|--------:|-----:|-------:|",
            ])
            for item in lines_items:
                lines.append(
                    f"| {item.get('description', '-')} "
                    f"| {self._fmt(item.get('quantity', 0), 2)} "
                    f"| {self._format_currency(item.get('rate', 0))} "
                    f"| {self._format_currency(item.get('amount', 0))} |"
                )
            total = sum(item.get("amount", 0) for item in lines_items)
            lines.append(
                f"| **TOTAL SETTLEMENT** | | | **{self._format_currency(total)}** |"
            )
        return "\n".join(lines)

    def _md_certification(self, data: Dict[str, Any]) -> str:
        """Render certification and attestation section."""
        cert = data.get("certification", {})
        lines = [
            "## 7. Certification & Attestation\n",
            f"- **Prepared By:** {cert.get('prepared_by', '-')}",
            f"- **Title:** {cert.get('title', '-')}",
            f"- **Date:** {cert.get('date', '-')}",
            f"- **Reviewed By:** {cert.get('reviewed_by', '-')}",
            f"- **Review Date:** {cert.get('review_date', '-')}",
            f"- **Data Integrity Hash:** `{cert.get('data_integrity_hash', '-')}`",
            "",
            "I hereby certify that the meter data, baseline calculations, and "
            "settlement amounts presented in this report are accurate, complete, "
            "and comply with the applicable program rules and tariff provisions.",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Settlement Verification Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Account: {data.get("account_id", "-")} | '
            f'Event: {data.get("event_id", "-")} | '
            f'Meter: {data.get("meter_id", "-")}</p>'
        )

    def _html_settlement_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML settlement summary cards."""
        s = data.get("settlement_summary", {})
        return (
            '<h2>Settlement Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CBL Average</span>'
            f'<span class="value">{self._fmt(s.get("cbl_average_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Actual Average</span>'
            f'<span class="value">{self._fmt(s.get("actual_average_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Verified Curtailment</span>'
            f'<span class="value">{self._fmt(s.get("verified_curtailment_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Performance Factor</span>'
            f'<span class="value">{self._fmt(s.get("performance_factor", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">Settlement Amount</span>'
            f'<span class="value">{self._format_currency(s.get("settlement_amount", 0))}</span></div>\n'
            '</div>'
        )

    def _html_baseline_documentation(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline documentation."""
        baseline = data.get("baseline_documentation", {})
        days = baseline.get("baseline_days", [])
        rows = ""
        for d in days:
            cls = "" if d.get("included", True) else "row-excluded"
            rows += (
                f'<tr class="{cls}"><td>{d.get("date", "-")}</td>'
                f'<td>{d.get("day_type", "-")}</td>'
                f'<td>{self._fmt(d.get("peak_kw", 0), 1)}</td>'
                f'<td>{"Yes" if d.get("included", True) else "No"}</td></tr>\n'
            )
        return (
            '<h2>Baseline Documentation</h2>\n'
            f'<p>Methodology: {baseline.get("methodology", "-")} | '
            f'Adjustment: {baseline.get("adjustment_method", "None")}</p>\n'
            '<table>\n<tr><th>Date</th><th>Day Type</th>'
            f'<th>Peak kW</th><th>Included</th></tr>\n{rows}</table>'
        )

    def _html_interval_meter_data(self, data: Dict[str, Any]) -> str:
        """Render HTML interval meter data table."""
        intervals = data.get("interval_meter_data", {}).get("intervals", [])
        rows = ""
        for iv in intervals:
            baseline_kw = iv.get("baseline_kw", 0)
            actual_kw = iv.get("actual_kw", 0)
            curtailment = baseline_kw - actual_kw
            rows += (
                f'<tr><td>{iv.get("interval", "-")}</td>'
                f'<td>{self._fmt(baseline_kw, 1)}</td>'
                f'<td>{self._fmt(actual_kw, 1)}</td>'
                f'<td>{self._fmt(curtailment, 1)}</td></tr>\n'
            )
        return (
            '<h2>Interval Meter Data</h2>\n'
            '<table>\n<tr><th>Interval</th><th>Baseline kW</th>'
            f'<th>Actual kW</th><th>Curtailment kW</th></tr>\n{rows}</table>'
        )

    def _html_performance_verification(self, data: Dict[str, Any]) -> str:
        """Render HTML performance verification."""
        perf = data.get("performance_verification", {})
        status_cls = "status-pass" if perf.get("compliance_status") == "Compliant" else "status-fail"
        return (
            '<h2>Performance Verification</h2>\n'
            f'<div class="{status_cls}">'
            f'<strong>Status: {perf.get("compliance_status", "-")}</strong> | '
            f'Committed: {self._fmt(perf.get("committed_capacity_kw", 0), 0)} kW | '
            f'Verified: {self._fmt(perf.get("verified_curtailment_kw", 0), 0)} kW | '
            f'Ratio: {self._fmt(perf.get("performance_ratio", 0), 4)}</div>'
        )

    def _html_adjustment_documentation(self, data: Dict[str, Any]) -> str:
        """Render HTML adjustment documentation table."""
        adjustments = data.get("adjustment_documentation", [])
        rows = ""
        for adj in adjustments:
            rows += (
                f'<tr><td>{adj.get("type", "-")}</td>'
                f'<td>{self._fmt(adj.get("factor", 1.0), 4)}</td>'
                f'<td>{self._fmt(adj.get("impact_kw", 0), 1)}</td>'
                f'<td>{adj.get("justification", "-")}</td></tr>\n'
            )
        return (
            '<h2>Adjustments</h2>\n'
            '<table>\n<tr><th>Type</th><th>Factor</th>'
            f'<th>Impact kW</th><th>Justification</th></tr>\n{rows}</table>'
        )

    def _html_settlement_calculations(self, data: Dict[str, Any]) -> str:
        """Render HTML settlement calculations."""
        calc = data.get("settlement_calculations", {})
        items = calc.get("line_items", [])
        rows = ""
        for item in items:
            rows += (
                f'<tr><td>{item.get("description", "-")}</td>'
                f'<td>{self._fmt(item.get("quantity", 0), 2)}</td>'
                f'<td>{self._format_currency(item.get("rate", 0))}</td>'
                f'<td>{self._format_currency(item.get("amount", 0))}</td></tr>\n'
            )
        total = sum(item.get("amount", 0) for item in items)
        rows += f'<tr class="total-row"><td><strong>TOTAL</strong></td><td></td><td></td><td><strong>{self._format_currency(total)}</strong></td></tr>\n'
        return (
            '<h2>Settlement Calculations</h2>\n'
            f'<p>Price Node: {calc.get("price_node", "-")} | '
            f'Settlement Price: {self._format_currency(calc.get("settlement_price_per_kwh", 0))}/kWh</p>\n'
            '<table>\n<tr><th>Line Item</th><th>Quantity</th>'
            f'<th>Rate</th><th>Amount</th></tr>\n{rows}</table>'
        )

    def _html_certification(self, data: Dict[str, Any]) -> str:
        """Render HTML certification."""
        cert = data.get("certification", {})
        return (
            '<h2>Certification &amp; Attestation</h2>\n'
            f'<div class="certification-box">'
            f'<p><strong>Prepared By:</strong> {cert.get("prepared_by", "-")} | '
            f'<strong>Date:</strong> {cert.get("date", "-")}</p>'
            f'<p><strong>Reviewed By:</strong> {cert.get("reviewed_by", "-")} | '
            f'<strong>Review Date:</strong> {cert.get("review_date", "-")}</p>'
            f'<p><strong>Data Integrity Hash:</strong> <code>{cert.get("data_integrity_hash", "-")}</code></p>'
            f'<p><em>I hereby certify that the meter data, baseline calculations, '
            f'and settlement amounts presented in this report are accurate and complete.</em></p>'
            f'</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_settlement_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON settlement summary."""
        s = data.get("settlement_summary", {})
        return {
            "event_window": s.get("event_window", ""),
            "duration_hours": s.get("duration_hours", 0),
            "baseline_methodology": s.get("baseline_methodology", ""),
            "cbl_average_kw": s.get("cbl_average_kw", 0),
            "actual_average_kw": s.get("actual_average_kw", 0),
            "verified_curtailment_kw": s.get("verified_curtailment_kw", 0),
            "committed_curtailment_kw": s.get("committed_curtailment_kw", 0),
            "performance_factor": s.get("performance_factor", 0),
            "settlement_amount": s.get("settlement_amount", 0),
            "settlement_status": s.get("settlement_status", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        intervals = data.get("interval_meter_data", {}).get("intervals", [])
        items = data.get("settlement_calculations", {}).get("line_items", [])
        return {
            "baseline_actual_line": {
                "type": "line",
                "labels": [iv.get("interval", "") for iv in intervals],
                "series": {
                    "baseline": [iv.get("baseline_kw", 0) for iv in intervals],
                    "actual": [iv.get("actual_kw", 0) for iv in intervals],
                },
            },
            "curtailment_area": {
                "type": "area",
                "labels": [iv.get("interval", "") for iv in intervals],
                "values": [
                    iv.get("baseline_kw", 0) - iv.get("actual_kw", 0)
                    for iv in intervals
                ],
            },
            "settlement_breakdown": {
                "type": "bar",
                "labels": [item.get("description", "") for item in items],
                "values": [item.get("amount", 0) for item in items],
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
            ".row-excluded{background:#f8d7da !important;}"
            ".total-row{background:#e2e3f1 !important;font-weight:700;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-pass{background:#d1e7dd;padding:15px;border-radius:8px;margin:10px 0;}"
            ".status-fail{background:#f8d7da;padding:15px;border-radius:8px;margin:10px 0;}"
            ".certification-box{background:#f0f9ff;border:2px solid #0d6efd;padding:20px;border-radius:8px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
