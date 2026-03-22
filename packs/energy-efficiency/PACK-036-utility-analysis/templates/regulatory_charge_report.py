# -*- coding: utf-8 -*-
"""
RegulatoryChargeReportTemplate - Regulatory surcharge analysis report for PACK-036.

Generates regulatory charge reports covering itemized charge decomposition,
network and capacity charge analysis, tax and levy assessments, exemption
eligibility reviews, optimization opportunities, and cost projections
for future regulatory periods. Designed for energy managers and
regulatory compliance teams managing non-commodity utility charges.

Sections:
    1. Header & Charge Summary
    2. Charge Decomposition
    3. Network Charges
    4. Taxes & Levies
    5. Exemption Analysis
    6. Optimization Opportunities
    7. Projections
    8. Provenance

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash excluding volatile fields."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {
            k: v for k, v in s.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


class RegulatoryChargeReportTemplate:
    """
    Regulatory surcharge analysis report template.

    Renders regulatory charge analysis including charge decomposition,
    network charges, tax/levy breakdowns, exemption eligibility,
    optimization opportunities, and future cost projections across
    markdown, HTML, JSON, and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryChargeReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render regulatory charge report as Markdown.

        Args:
            data: Regulatory charge data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_charge_decomposition(data),
            self._md_network_charges(data),
            self._md_taxes_levies(data),
            self._md_exemptions(data),
            self._md_optimization(data),
            self._md_projections(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory charge report as self-contained HTML.

        Args:
            data: Regulatory charge data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_charge_decomposition(data),
            self._html_network_charges(data),
            self._html_taxes_levies(data),
            self._html_exemptions(data),
            self._html_optimization(data),
            self._html_projections(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulatory Charge Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory charge report as structured JSON.

        Args:
            data: Regulatory charge data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_charge_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "charge_summary": data.get("charge_summary", {}),
            "charge_decomposition": data.get("charge_decomposition", []),
            "network_charges": data.get("network_charges", {}),
            "taxes_levies": data.get("taxes_levies", []),
            "exemptions": data.get("exemptions", []),
            "optimization": data.get("optimization", []),
            "projections": data.get("projections", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render charge decomposition as CSV.

        Args:
            data: Regulatory charge data from engine processing.

        Returns:
            CSV string with one row per charge component.
        """
        self.generated_at = _utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Charge Category", "Charge Name", "Rate", "Units",
            "Monthly Amount", "Annual Amount", "Share (%)",
            "Controllable", "Optimization Potential",
        ])
        for charge in data.get("charge_decomposition", []):
            writer.writerow([
                charge.get("category", ""),
                charge.get("name", ""),
                self._fmt_raw(charge.get("rate", 0), 4),
                charge.get("units", ""),
                self._fmt_raw(charge.get("monthly_amount", 0)),
                self._fmt_raw(charge.get("annual_amount", 0)),
                self._fmt_raw(charge.get("share_pct", 0)),
                charge.get("controllable", ""),
                self._fmt_raw(charge.get("optimization_potential", 0)),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with charge summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("charge_summary", {})
        return (
            "# Regulatory Charge Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Account:** {data.get('account_number', '-')}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '-')}  \n"
            f"**Total Regulatory Charges:** {self._fmt_currency(summary.get('total_regulatory', 0))}  \n"
            f"**Share of Total Bill:** {self._fmt(summary.get('share_of_bill_pct', 0))}%  \n"
            f"**Optimization Potential:** {self._fmt_currency(summary.get('optimization_potential', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 RegulatoryChargeReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_charge_decomposition(self, data: Dict[str, Any]) -> str:
        """Render charge decomposition section."""
        charges = data.get("charge_decomposition", [])
        if not charges:
            return "## 1. Charge Decomposition\n\n_No charge data available._"
        lines = [
            "## 1. Charge Decomposition\n",
            "| Category | Charge | Rate | Annual Amount | Share (%) | Controllable |",
            "|----------|--------|------|-------------|----------|-------------|",
        ]
        for c in charges:
            ctrl = "Yes" if c.get("controllable", False) else "No"
            lines.append(
                f"| {c.get('category', '-')} "
                f"| {c.get('name', '-')} "
                f"| {self._fmt(c.get('rate', 0), 4)} {c.get('units', '')} "
                f"| {self._fmt_currency(c.get('annual_amount', 0))} "
                f"| {self._fmt(c.get('share_pct', 0))}% "
                f"| {ctrl} |"
            )
        return "\n".join(lines)

    def _md_network_charges(self, data: Dict[str, Any]) -> str:
        """Render network charges section."""
        nc = data.get("network_charges", {})
        components = nc.get("components", [])
        lines = [
            "## 2. Network Charges\n",
            f"**Distribution Charges:** {self._fmt_currency(nc.get('distribution_annual', 0))} /yr  ",
            f"**Transmission Charges:** {self._fmt_currency(nc.get('transmission_annual', 0))} /yr  ",
            f"**Capacity Charges:** {self._fmt_currency(nc.get('capacity_annual', 0))} /yr  ",
            f"**Total Network:** {self._fmt_currency(nc.get('total_network', 0))} /yr  ",
            f"**Network Share of Bill:** {self._fmt(nc.get('share_pct', 0))}%\n",
        ]
        if components:
            lines.extend([
                "| Component | Basis | Rate | Annual Cost |",
                "|-----------|-------|------|-----------|",
            ])
            for comp in components:
                lines.append(
                    f"| {comp.get('name', '-')} "
                    f"| {comp.get('basis', '-')} "
                    f"| {self._fmt(comp.get('rate', 0), 4)} "
                    f"| {self._fmt_currency(comp.get('annual_cost', 0))} |"
                )
        return "\n".join(lines)

    def _md_taxes_levies(self, data: Dict[str, Any]) -> str:
        """Render taxes and levies section."""
        taxes = data.get("taxes_levies", [])
        if not taxes:
            return "## 3. Taxes & Levies\n\n_No tax data available._"
        lines = [
            "## 3. Taxes & Levies\n",
            "| Tax / Levy | Rate | Basis | Annual Amount | Exempt | Notes |",
            "|-----------|------|-------|-------------|--------|-------|",
        ]
        for t in taxes:
            exempt = "Yes" if t.get("exempt", False) else "No"
            lines.append(
                f"| {t.get('name', '-')} "
                f"| {self._fmt(t.get('rate', 0), 4)} "
                f"| {t.get('basis', '-')} "
                f"| {self._fmt_currency(t.get('annual_amount', 0))} "
                f"| {exempt} "
                f"| {t.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_exemptions(self, data: Dict[str, Any]) -> str:
        """Render exemption analysis section."""
        exemptions = data.get("exemptions", [])
        if not exemptions:
            return "## 4. Exemption Analysis\n\n_No exemptions identified._"
        lines = [
            "## 4. Exemption Analysis\n",
            "| Exemption | Eligible | Annual Savings | Requirement | Status |",
            "|-----------|---------|---------------|-------------|--------|",
        ]
        for e in exemptions:
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {e.get('eligible', '-')} "
                f"| {self._fmt_currency(e.get('annual_savings', 0))} "
                f"| {e.get('requirement', '-')} "
                f"| {e.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_optimization(self, data: Dict[str, Any]) -> str:
        """Render optimization opportunities section."""
        opts = data.get("optimization", [])
        if not opts:
            return "## 5. Optimization Opportunities\n\n_No optimization opportunities identified._"
        lines = [
            "## 5. Optimization Opportunities\n",
            "| # | Opportunity | Category | Annual Savings | Effort | Payback |",
            "|---|-----------|----------|---------------|--------|---------|",
        ]
        for i, o in enumerate(opts, 1):
            lines.append(
                f"| {i} | {o.get('opportunity', '-')} "
                f"| {o.get('category', '-')} "
                f"| {self._fmt_currency(o.get('annual_savings', 0))} "
                f"| {o.get('effort', '-')} "
                f"| {o.get('payback', '-')} |"
            )
        return "\n".join(lines)

    def _md_projections(self, data: Dict[str, Any]) -> str:
        """Render future regulatory charge projections."""
        projections = data.get("projections", [])
        if not projections:
            return "## 6. Projections\n\n_No projections available._"
        lines = [
            "## 6. Regulatory Charge Projections\n",
            "| Period | Network | Taxes/Levies | Other | Total | Change (%) |",
            "|--------|---------|------------|-------|-------|-----------|",
        ]
        for p in projections:
            lines.append(
                f"| {p.get('period', '-')} "
                f"| {self._fmt_currency(p.get('network', 0))} "
                f"| {self._fmt_currency(p.get('taxes_levies', 0))} "
                f"| {self._fmt_currency(p.get('other', 0))} "
                f"| {self._fmt_currency(p.get('total', 0))} "
                f"| {self._fmt(p.get('change_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Regulatory charge analysis based on published tariff schedules. "
            "Exemption eligibility requires formal application to regulator.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with charge summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("charge_summary", {})
        return (
            f'<h1>Regulatory Charge Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Account: {data.get("account_number", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Regulatory</span>'
            f'<span class="value">{self._fmt_currency(summary.get("total_regulatory", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Share of Bill</span>'
            f'<span class="value">{self._fmt(summary.get("share_of_bill_pct", 0))}%</span></div>\n'
            f'  <div class="card card-green"><span class="label">Optimization</span>'
            f'<span class="value">{self._fmt_currency(summary.get("optimization_potential", 0))}</span></div>\n'
            f'</div>'
        )

    def _html_charge_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML charge decomposition table."""
        charges = data.get("charge_decomposition", [])
        rows = ""
        for c in charges:
            ctrl = "Yes" if c.get("controllable", False) else "No"
            rows += (
                f'<tr><td>{c.get("category", "-")}</td>'
                f'<td>{c.get("name", "-")}</td>'
                f'<td>{self._fmt(c.get("rate", 0), 4)}</td>'
                f'<td>{self._fmt_currency(c.get("annual_amount", 0))}</td>'
                f'<td>{self._fmt(c.get("share_pct", 0))}%</td>'
                f'<td>{ctrl}</td></tr>\n'
            )
        return (
            '<h2>Charge Decomposition</h2>\n'
            '<table>\n<tr><th>Category</th><th>Charge</th><th>Rate</th>'
            '<th>Annual Amount</th><th>Share</th>'
            f'<th>Controllable</th></tr>\n{rows}</table>'
        )

    def _html_network_charges(self, data: Dict[str, Any]) -> str:
        """Render HTML network charges section."""
        nc = data.get("network_charges", {})
        components = nc.get("components", [])
        rows = ""
        for comp in components:
            rows += (
                f'<tr><td>{comp.get("name", "-")}</td>'
                f'<td>{comp.get("basis", "-")}</td>'
                f'<td>{self._fmt(comp.get("rate", 0), 4)}</td>'
                f'<td>{self._fmt_currency(comp.get("annual_cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Network Charges</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Distribution</span>'
            f'<span class="value">{self._fmt_currency(nc.get("distribution_annual", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Transmission</span>'
            f'<span class="value">{self._fmt_currency(nc.get("transmission_annual", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Capacity</span>'
            f'<span class="value">{self._fmt_currency(nc.get("capacity_annual", 0))}</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Component</th><th>Basis</th>'
            f'<th>Rate</th><th>Annual Cost</th></tr>\n{rows}</table>'
        )

    def _html_taxes_levies(self, data: Dict[str, Any]) -> str:
        """Render HTML taxes and levies section."""
        taxes = data.get("taxes_levies", [])
        rows = ""
        for t in taxes:
            exempt = "Yes" if t.get("exempt", False) else "No"
            rows += (
                f'<tr><td>{t.get("name", "-")}</td>'
                f'<td>{self._fmt(t.get("rate", 0), 4)}</td>'
                f'<td>{t.get("basis", "-")}</td>'
                f'<td>{self._fmt_currency(t.get("annual_amount", 0))}</td>'
                f'<td>{exempt}</td></tr>\n'
            )
        return (
            '<h2>Taxes & Levies</h2>\n'
            '<table>\n<tr><th>Tax / Levy</th><th>Rate</th><th>Basis</th>'
            f'<th>Annual Amount</th><th>Exempt</th></tr>\n{rows}</table>'
        )

    def _html_exemptions(self, data: Dict[str, Any]) -> str:
        """Render HTML exemption analysis section."""
        exemptions = data.get("exemptions", [])
        rows = ""
        for e in exemptions:
            cls = "card-green" if e.get("eligible") == "Yes" else ""
            rows += (
                f'<tr class="{cls}"><td>{e.get("name", "-")}</td>'
                f'<td>{e.get("eligible", "-")}</td>'
                f'<td>{self._fmt_currency(e.get("annual_savings", 0))}</td>'
                f'<td>{e.get("requirement", "-")}</td>'
                f'<td>{e.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Exemption Analysis</h2>\n'
            '<table>\n<tr><th>Exemption</th><th>Eligible</th>'
            '<th>Annual Savings</th><th>Requirement</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_optimization(self, data: Dict[str, Any]) -> str:
        """Render HTML optimization opportunities section."""
        opts = data.get("optimization", [])
        rows = ""
        for o in opts:
            rows += (
                f'<tr><td>{o.get("opportunity", "-")}</td>'
                f'<td>{o.get("category", "-")}</td>'
                f'<td>{self._fmt_currency(o.get("annual_savings", 0))}</td>'
                f'<td>{o.get("effort", "-")}</td>'
                f'<td>{o.get("payback", "-")}</td></tr>\n'
            )
        return (
            '<h2>Optimization Opportunities</h2>\n'
            '<table>\n<tr><th>Opportunity</th><th>Category</th>'
            '<th>Annual Savings</th><th>Effort</th>'
            f'<th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML regulatory charge projections."""
        projections = data.get("projections", [])
        rows = ""
        for p in projections:
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{self._fmt_currency(p.get("network", 0))}</td>'
                f'<td>{self._fmt_currency(p.get("taxes_levies", 0))}</td>'
                f'<td>{self._fmt_currency(p.get("total", 0))}</td>'
                f'<td>{self._fmt(p.get("change_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Regulatory Charge Projections</h2>\n'
            '<table>\n<tr><th>Period</th><th>Network</th>'
            '<th>Taxes/Levies</th><th>Total</th>'
            f'<th>Change (%)</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        charges = data.get("charge_decomposition", [])
        projections = data.get("projections", [])
        return {
            "charge_breakdown_pie": {
                "type": "pie",
                "labels": [c.get("name", "") for c in charges],
                "values": [c.get("annual_amount", 0) for c in charges],
            },
            "projection_line": {
                "type": "line",
                "labels": [p.get("period", "") for p in projections],
                "series": {
                    "network": [p.get("network", 0) for p in projections],
                    "taxes_levies": [p.get("taxes_levies", 0) for p in projections],
                    "total": [p.get("total", 0) for p in projections],
                },
            },
            "category_bar": {
                "type": "bar",
                "labels": list({c.get("category", "") for c in charges}),
                "values": self._aggregate_by_category(charges),
            },
        }

    def _aggregate_by_category(self, charges: List[Dict[str, Any]]) -> List[float]:
        """Aggregate charge amounts by category for chart data."""
        cat_totals: Dict[str, float] = {}
        for c in charges:
            cat = c.get("category", "Other")
            cat_totals[cat] = cat_totals.get(cat, 0) + c.get("annual_amount", 0)
        return list(cat_totals.values())

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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _fmt_raw(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value without commas (for CSV)."""
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    def _fmt_currency(self, val: Any, symbol: str = "") -> str:
        """Format a currency value."""
        sym = symbol or self.config.get("currency_symbol", "EUR")
        if isinstance(val, (int, float)):
            return f"{sym} {val:,.2f}"
        return f"{sym} {val}"

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage."""
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
