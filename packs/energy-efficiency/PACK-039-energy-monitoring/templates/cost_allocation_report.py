# -*- coding: utf-8 -*-
"""
CostAllocationReportTemplate - Cost allocation for PACK-039.

Generates comprehensive energy cost allocation reports showing cost
distribution by tenant and department with metered consumption data,
demand contribution analysis, reconciliation to utility bills, and
charge-back summary with variance analysis.

Sections:
    1. Allocation Overview
    2. Tenant / Department Allocation
    3. Demand Contribution
    4. Rate Analysis
    5. Reconciliation to Utility Bill
    6. Charge-Back Summary
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management systems - Cost allocation)
    - ASHRAE Standard 105 (Standard methods of determining allocation)
    - IPMVP Option C (Whole-facility metering approach)

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


class CostAllocationReportTemplate:
    """
    Cost allocation report template.

    Renders energy cost allocation reports showing distribution by tenant
    and department with metered consumption, demand contribution analysis,
    utility bill reconciliation, and charge-back summaries across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CostAllocationReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render cost allocation report as Markdown.

        Args:
            data: Cost allocation engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_allocation_overview(data),
            self._md_tenant_allocation(data),
            self._md_demand_contribution(data),
            self._md_rate_analysis(data),
            self._md_reconciliation(data),
            self._md_chargeback_summary(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render cost allocation report as self-contained HTML.

        Args:
            data: Cost allocation engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_allocation_overview(data),
            self._html_tenant_allocation(data),
            self._html_demand_contribution(data),
            self._html_rate_analysis(data),
            self._html_reconciliation(data),
            self._html_chargeback_summary(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Cost Allocation Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render cost allocation report as structured JSON.

        Args:
            data: Cost allocation engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "cost_allocation_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "allocation_overview": self._json_allocation_overview(data),
            "tenant_allocation": data.get("tenant_allocation", []),
            "demand_contribution": data.get("demand_contribution", []),
            "rate_analysis": data.get("rate_analysis", {}),
            "reconciliation": data.get("reconciliation", {}),
            "chargeback_summary": data.get("chargeback_summary", []),
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
            f"# Cost Allocation Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Billing Period:** {data.get('billing_period', '')}  \n"
            f"**Total Cost:** {self._format_currency(data.get('total_cost', 0))}  \n"
            f"**Total Consumption:** {self._format_energy(data.get('total_consumption_mwh', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 CostAllocationReportTemplate v39.0.0\n\n---"
        )

    def _md_allocation_overview(self, data: Dict[str, Any]) -> str:
        """Render allocation overview section."""
        overview = data.get("allocation_overview", {})
        total_cost = data.get("total_cost", 0)
        return (
            "## 1. Allocation Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Utility Cost | {self._format_currency(total_cost)} |\n"
            f"| Energy Charges | {self._format_currency(overview.get('energy_charges', 0))} |\n"
            f"| Demand Charges | {self._format_currency(overview.get('demand_charges', 0))} |\n"
            f"| Fixed Charges | {self._format_currency(overview.get('fixed_charges', 0))} |\n"
            f"| Taxes & Fees | {self._format_currency(overview.get('taxes_fees', 0))} |\n"
            f"| Tenants / Departments | {self._fmt(overview.get('tenant_count', 0), 0)} |\n"
            f"| Metered Coverage | {self._fmt(overview.get('metered_coverage_pct', 0))}% |\n"
            f"| Unallocated Amount | {self._format_currency(overview.get('unallocated', 0))} |"
        )

    def _md_tenant_allocation(self, data: Dict[str, Any]) -> str:
        """Render tenant / department allocation section."""
        tenants = data.get("tenant_allocation", [])
        if not tenants:
            return "## 2. Tenant / Department Allocation\n\n_No tenant allocation data available._"
        total_cost = data.get("total_cost", 1)
        lines = [
            "## 2. Tenant / Department Allocation\n",
            "| Tenant / Dept | Consumption (MWh) | % of Total | Energy Cost | Demand Cost | Total Cost |",
            "|--------------|------------------:|----------:|-----------:|-----------:|-----------:|",
        ]
        for t in tenants:
            cost = t.get("total_cost", 0)
            lines.append(
                f"| {t.get('name', '-')} "
                f"| {self._fmt(t.get('consumption_mwh', 0), 1)} "
                f"| {self._pct(cost, total_cost)} "
                f"| {self._format_currency(t.get('energy_cost', 0))} "
                f"| {self._format_currency(t.get('demand_cost', 0))} "
                f"| {self._format_currency(cost)} |"
            )
        return "\n".join(lines)

    def _md_demand_contribution(self, data: Dict[str, Any]) -> str:
        """Render demand contribution section."""
        demands = data.get("demand_contribution", [])
        if not demands:
            return "## 3. Demand Contribution\n\n_No demand contribution data available._"
        lines = [
            "## 3. Demand Contribution\n",
            "| Tenant / Dept | Peak kW | % of System Peak | Coincident kW | Diversity Factor |",
            "|--------------|-------:|----------------:|-------------:|---------------:|",
        ]
        for d in demands:
            lines.append(
                f"| {d.get('name', '-')} "
                f"| {self._fmt(d.get('peak_kw', 0), 1)} "
                f"| {self._fmt(d.get('pct_of_system_peak', 0))}% "
                f"| {self._fmt(d.get('coincident_kw', 0), 1)} "
                f"| {self._fmt(d.get('diversity_factor', 0), 3)} |"
            )
        return "\n".join(lines)

    def _md_rate_analysis(self, data: Dict[str, Any]) -> str:
        """Render rate analysis section."""
        rate = data.get("rate_analysis", {})
        if not rate:
            return "## 4. Rate Analysis\n\n_No rate analysis data available._"
        tiers = rate.get("tiers", [])
        lines = [
            "## 4. Rate Analysis\n",
            f"**Tariff:** {rate.get('tariff_name', '-')}  \n"
            f"**Rate Schedule:** {rate.get('rate_schedule', '-')}  \n"
            f"**Blended Rate:** {self._format_currency(rate.get('blended_rate', 0))}/MWh\n",
        ]
        if tiers:
            lines.append("### Rate Tiers\n")
            lines.append("| Tier | Range (kWh) | Rate (EUR/kWh) | Consumption (kWh) | Cost |")
            lines.append("|------|----------:|-------------:|------------------:|-----:|")
            for tier in tiers:
                lines.append(
                    f"| {tier.get('tier', '-')} "
                    f"| {tier.get('range', '-')} "
                    f"| {self._fmt(tier.get('rate', 0), 4)} "
                    f"| {self._fmt(tier.get('consumption_kwh', 0), 0)} "
                    f"| {self._format_currency(tier.get('cost', 0))} |"
                )
        return "\n".join(lines)

    def _md_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render reconciliation to utility bill section."""
        recon = data.get("reconciliation", {})
        if not recon:
            return "## 5. Reconciliation to Utility Bill\n\n_No reconciliation data available._"
        line_items = recon.get("line_items", [])
        lines = [
            "## 5. Reconciliation to Utility Bill\n",
            f"**Bill Amount:** {self._format_currency(recon.get('bill_amount', 0))}  \n"
            f"**Allocated Amount:** {self._format_currency(recon.get('allocated_amount', 0))}  \n"
            f"**Variance:** {self._format_currency(recon.get('variance', 0))}  \n"
            f"**Variance %:** {self._fmt(recon.get('variance_pct', 0))}%\n",
        ]
        if line_items:
            lines.append("### Line Item Reconciliation\n")
            lines.append("| Line Item | Bill | Allocated | Variance |")
            lines.append("|-----------|-----:|----------:|---------:|")
            for li in line_items:
                lines.append(
                    f"| {li.get('item', '-')} "
                    f"| {self._format_currency(li.get('bill_amount', 0))} "
                    f"| {self._format_currency(li.get('allocated_amount', 0))} "
                    f"| {self._format_currency(li.get('variance', 0))} |"
                )
        return "\n".join(lines)

    def _md_chargeback_summary(self, data: Dict[str, Any]) -> str:
        """Render charge-back summary section."""
        chargebacks = data.get("chargeback_summary", [])
        if not chargebacks:
            return "## 6. Charge-Back Summary\n\n_No charge-back data available._"
        lines = [
            "## 6. Charge-Back Summary\n",
            "| Tenant / Dept | Chargeback Amount | Adjustment | Final Amount | Status |",
            "|--------------|------------------:|-----------:|------------:|--------|",
        ]
        for cb in chargebacks:
            lines.append(
                f"| {cb.get('name', '-')} "
                f"| {self._format_currency(cb.get('chargeback_amount', 0))} "
                f"| {self._format_currency(cb.get('adjustment', 0))} "
                f"| {self._format_currency(cb.get('final_amount', 0))} "
                f"| {cb.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Install sub-meters for unmetered tenants to improve allocation accuracy",
                "Review demand allocation methodology for fairness and transparency",
                "Reconcile metered data with utility bills monthly to detect drift",
                "Implement automated charge-back invoicing to reduce manual effort",
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
            f'<h1>Cost Allocation Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Total Cost: {self._format_currency(data.get("total_cost", 0))} | '
            f'Period: {data.get("billing_period", "-")}</p>'
        )

    def _html_allocation_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML allocation overview cards."""
        o = data.get("allocation_overview", {})
        return (
            '<h2>Allocation Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">{self._format_currency(data.get("total_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Energy Charges</span>'
            f'<span class="value">{self._format_currency(o.get("energy_charges", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Demand Charges</span>'
            f'<span class="value">{self._format_currency(o.get("demand_charges", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Metered Coverage</span>'
            f'<span class="value">{self._fmt(o.get("metered_coverage_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Tenants</span>'
            f'<span class="value">{self._fmt(o.get("tenant_count", 0), 0)}</span></div>\n'
            '</div>'
        )

    def _html_tenant_allocation(self, data: Dict[str, Any]) -> str:
        """Render HTML tenant allocation table."""
        tenants = data.get("tenant_allocation", [])
        rows = ""
        for t in tenants:
            rows += (
                f'<tr><td>{t.get("name", "-")}</td>'
                f'<td>{self._fmt(t.get("consumption_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(t.get("energy_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("demand_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("total_cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Tenant / Department Allocation</h2>\n'
            '<table>\n<tr><th>Tenant / Dept</th><th>Consumption (MWh)</th>'
            f'<th>Energy Cost</th><th>Demand Cost</th><th>Total Cost</th></tr>\n{rows}</table>'
        )

    def _html_demand_contribution(self, data: Dict[str, Any]) -> str:
        """Render HTML demand contribution table."""
        demands = data.get("demand_contribution", [])
        rows = ""
        for d in demands:
            rows += (
                f'<tr><td>{d.get("name", "-")}</td>'
                f'<td>{self._fmt(d.get("peak_kw", 0), 1)}</td>'
                f'<td>{self._fmt(d.get("pct_of_system_peak", 0))}%</td>'
                f'<td>{self._fmt(d.get("coincident_kw", 0), 1)}</td>'
                f'<td>{self._fmt(d.get("diversity_factor", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>Demand Contribution</h2>\n'
            '<table>\n<tr><th>Tenant / Dept</th><th>Peak kW</th>'
            '<th>% System Peak</th><th>Coincident kW</th>'
            f'<th>Diversity Factor</th></tr>\n{rows}</table>'
        )

    def _html_rate_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML rate analysis table."""
        rate = data.get("rate_analysis", {})
        tiers = rate.get("tiers", [])
        rows = ""
        for tier in tiers:
            rows += (
                f'<tr><td>{tier.get("tier", "-")}</td>'
                f'<td>{tier.get("range", "-")}</td>'
                f'<td>{self._fmt(tier.get("rate", 0), 4)}</td>'
                f'<td>{self._fmt(tier.get("consumption_kwh", 0), 0)}</td>'
                f'<td>{self._format_currency(tier.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Rate Analysis</h2>\n'
            f'<p>Tariff: {rate.get("tariff_name", "-")} | '
            f'Blended Rate: {self._format_currency(rate.get("blended_rate", 0))}/MWh</p>\n'
            '<table>\n<tr><th>Tier</th><th>Range (kWh)</th><th>Rate (EUR/kWh)</th>'
            f'<th>Consumption (kWh)</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render HTML reconciliation table."""
        recon = data.get("reconciliation", {})
        line_items = recon.get("line_items", [])
        rows = ""
        for li in line_items:
            var = li.get("variance", 0)
            cls = "severity-high" if abs(var) > 100 else ""
            rows += (
                f'<tr><td>{li.get("item", "-")}</td>'
                f'<td>{self._format_currency(li.get("bill_amount", 0))}</td>'
                f'<td>{self._format_currency(li.get("allocated_amount", 0))}</td>'
                f'<td class="{cls}">{self._format_currency(var)}</td></tr>\n'
            )
        return (
            '<h2>Reconciliation to Utility Bill</h2>\n'
            f'<p>Variance: {self._format_currency(recon.get("variance", 0))} '
            f'({self._fmt(recon.get("variance_pct", 0))}%)</p>\n'
            '<table>\n<tr><th>Line Item</th><th>Bill</th>'
            f'<th>Allocated</th><th>Variance</th></tr>\n{rows}</table>'
        )

    def _html_chargeback_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML charge-back summary table."""
        chargebacks = data.get("chargeback_summary", [])
        rows = ""
        for cb in chargebacks:
            rows += (
                f'<tr><td>{cb.get("name", "-")}</td>'
                f'<td>{self._format_currency(cb.get("chargeback_amount", 0))}</td>'
                f'<td>{self._format_currency(cb.get("adjustment", 0))}</td>'
                f'<td>{self._format_currency(cb.get("final_amount", 0))}</td>'
                f'<td>{cb.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Charge-Back Summary</h2>\n'
            '<table>\n<tr><th>Tenant / Dept</th><th>Chargeback</th>'
            f'<th>Adjustment</th><th>Final Amount</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Install sub-meters for unmetered tenants to improve allocation accuracy",
            "Reconcile metered data with utility bills monthly to detect drift",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_allocation_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON allocation overview."""
        o = data.get("allocation_overview", {})
        return {
            "total_cost": data.get("total_cost", 0),
            "energy_charges": o.get("energy_charges", 0),
            "demand_charges": o.get("demand_charges", 0),
            "fixed_charges": o.get("fixed_charges", 0),
            "taxes_fees": o.get("taxes_fees", 0),
            "tenant_count": o.get("tenant_count", 0),
            "metered_coverage_pct": o.get("metered_coverage_pct", 0),
            "unallocated": o.get("unallocated", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        tenants = data.get("tenant_allocation", [])
        demands = data.get("demand_contribution", [])
        return {
            "cost_distribution": {
                "type": "pie",
                "labels": [t.get("name", "") for t in tenants],
                "values": [t.get("total_cost", 0) for t in tenants],
            },
            "consumption_distribution": {
                "type": "pie",
                "labels": [t.get("name", "") for t in tenants],
                "values": [t.get("consumption_mwh", 0) for t in tenants],
            },
            "demand_contribution": {
                "type": "stacked_bar",
                "labels": [d.get("name", "") for d in demands],
                "series": {
                    "coincident_kw": [d.get("coincident_kw", 0) for d in demands],
                    "non_coincident_kw": [
                        d.get("peak_kw", 0) - d.get("coincident_kw", 0)
                        for d in demands
                    ],
                },
            },
            "cost_breakdown": {
                "type": "bar",
                "labels": ["Energy", "Demand", "Fixed", "Taxes"],
                "values": [
                    data.get("allocation_overview", {}).get("energy_charges", 0),
                    data.get("allocation_overview", {}).get("demand_charges", 0),
                    data.get("allocation_overview", {}).get("fixed_charges", 0),
                    data.get("allocation_overview", {}).get("taxes_fees", 0),
                ],
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
