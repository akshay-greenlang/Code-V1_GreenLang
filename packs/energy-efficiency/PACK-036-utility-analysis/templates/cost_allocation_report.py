# -*- coding: utf-8 -*-
"""
CostAllocationReportTemplate - Cost allocation by entity report for PACK-036.

Generates cost allocation reports with meter-to-entity mapping,
consumption disaggregation by department/tenant/process, demand charge
allocation methodology, shared services apportionment, tenant chargeback
summaries, and variance analysis against budgeted allocations.

Sections:
    1. Header & Allocation Summary
    2. Allocation Methodology
    3. Cost by Entity
    4. Meter-to-Entity Mapping
    5. Reconciliation
    6. Tenant Invoices
    7. Variance Analysis
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"

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

class CostAllocationReportTemplate:
    """
    Cost allocation report template.

    Renders cost allocation results including entity-level cost
    breakdowns, allocation methodology documentation, meter-to-entity
    mapping, reconciliation summaries, and tenant invoice generation
    across markdown, HTML, JSON, and CSV formats.

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
            data: Cost allocation data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_cost_by_entity(data),
            self._md_meter_mapping(data),
            self._md_reconciliation(data),
            self._md_tenant_invoices(data),
            self._md_variance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render cost allocation report as self-contained HTML.

        Args:
            data: Cost allocation data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_methodology(data),
            self._html_cost_by_entity(data),
            self._html_meter_mapping(data),
            self._html_reconciliation(data),
            self._html_tenant_invoices(data),
            self._html_variance(data),
        ])
        prov = self._provenance(body)
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
            data: Cost allocation data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "cost_allocation_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "allocation_summary": data.get("allocation_summary", {}),
            "methodology": data.get("methodology", {}),
            "cost_by_entity": data.get("cost_by_entity", []),
            "meter_mapping": data.get("meter_mapping", []),
            "reconciliation": data.get("reconciliation", {}),
            "tenant_invoices": data.get("tenant_invoices", []),
            "variance_analysis": data.get("variance_analysis", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render cost allocation as CSV with one row per entity.

        Args:
            data: Cost allocation data from engine processing.

        Returns:
            CSV string with entity-level cost allocation data.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Entity Name", "Entity Type", "Meters", "Consumption (kWh)",
            "Demand (kW)", "Energy Cost", "Demand Cost", "Shared Cost",
            "Total Cost", "Cost Share (%)", "Budget", "Variance",
        ])
        for entity in data.get("cost_by_entity", []):
            writer.writerow([
                entity.get("name", ""),
                entity.get("type", ""),
                entity.get("meter_count", 0),
                self._fmt_raw(entity.get("consumption_kwh", 0), 0),
                self._fmt_raw(entity.get("demand_kw", 0)),
                self._fmt_raw(entity.get("energy_cost", 0)),
                self._fmt_raw(entity.get("demand_cost", 0)),
                self._fmt_raw(entity.get("shared_cost", 0)),
                self._fmt_raw(entity.get("total_cost", 0)),
                self._fmt_raw(entity.get("cost_share_pct", 0)),
                self._fmt_raw(entity.get("budget", 0)),
                self._fmt_raw(entity.get("variance", 0)),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with allocation summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("allocation_summary", {})
        return (
            "# Cost Allocation Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Property:** {data.get('property_name', '-')}  \n"
            f"**Billing Period:** {data.get('billing_period', '-')}  \n"
            f"**Total Utility Cost:** {self._fmt_currency(summary.get('total_cost', 0))}  \n"
            f"**Entities Allocated:** {summary.get('entity_count', 0)}  \n"
            f"**Meters Mapped:** {summary.get('meter_count', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 CostAllocationReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render allocation methodology section."""
        meth = data.get("methodology", {})
        rules = meth.get("allocation_rules", [])
        lines = [
            "## 1. Allocation Methodology\n",
            f"**Primary Method:** {meth.get('primary_method', '-')}  ",
            f"**Demand Allocation:** {meth.get('demand_method', '-')}  ",
            f"**Shared Services Method:** {meth.get('shared_services_method', '-')}  ",
            f"**Common Area Treatment:** {meth.get('common_area_treatment', '-')}  ",
            f"**Proration Basis:** {meth.get('proration_basis', '-')}\n",
        ]
        if rules:
            lines.append("### Allocation Rules\n")
            for r in rules:
                lines.append(f"- {r}")
        return "\n".join(lines)

    def _md_cost_by_entity(self, data: Dict[str, Any]) -> str:
        """Render cost by entity section."""
        entities = data.get("cost_by_entity", [])
        if not entities:
            return "## 2. Cost by Entity\n\n_No entities allocated._"
        lines = [
            "## 2. Cost by Entity\n",
            "| Entity | Type | kWh | Energy Cost | Demand Cost | Shared | Total | Share (%) |",
            "|--------|------|-----|------------|------------|--------|-------|----------|",
        ]
        for e in entities:
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {e.get('type', '-')} "
                f"| {self._fmt(e.get('consumption_kwh', 0), 0)} "
                f"| {self._fmt_currency(e.get('energy_cost', 0))} "
                f"| {self._fmt_currency(e.get('demand_cost', 0))} "
                f"| {self._fmt_currency(e.get('shared_cost', 0))} "
                f"| {self._fmt_currency(e.get('total_cost', 0))} "
                f"| {self._fmt(e.get('cost_share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_meter_mapping(self, data: Dict[str, Any]) -> str:
        """Render meter-to-entity mapping section."""
        mapping = data.get("meter_mapping", [])
        if not mapping:
            return "## 3. Meter-to-Entity Mapping\n\n_No meter mapping configured._"
        lines = [
            "## 3. Meter-to-Entity Mapping\n",
            "| Meter ID | Meter Name | Entity | Utility Type | Method |",
            "|----------|-----------|--------|-------------|--------|",
        ]
        for m in mapping:
            lines.append(
                f"| {m.get('meter_id', '-')} "
                f"| {m.get('meter_name', '-')} "
                f"| {m.get('entity', '-')} "
                f"| {m.get('utility_type', '-')} "
                f"| {m.get('allocation_method', '-')} |"
            )
        return "\n".join(lines)

    def _md_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render reconciliation section."""
        rec = data.get("reconciliation", {})
        line_items = rec.get("line_items", [])
        lines = [
            "## 4. Reconciliation\n",
            f"**Total Billed:** {self._fmt_currency(rec.get('total_billed', 0))}  ",
            f"**Total Allocated:** {self._fmt_currency(rec.get('total_allocated', 0))}  ",
            f"**Unallocated:** {self._fmt_currency(rec.get('unallocated', 0))}  ",
            f"**Reconciliation Status:** {rec.get('status', '-')}\n",
        ]
        if line_items:
            lines.extend([
                "| Line Item | Billed | Allocated | Difference |",
                "|-----------|--------|-----------|-----------|",
            ])
            for li in line_items:
                lines.append(
                    f"| {li.get('item', '-')} "
                    f"| {self._fmt_currency(li.get('billed', 0))} "
                    f"| {self._fmt_currency(li.get('allocated', 0))} "
                    f"| {self._fmt_currency(li.get('difference', 0))} |"
                )
        return "\n".join(lines)

    def _md_tenant_invoices(self, data: Dict[str, Any]) -> str:
        """Render tenant invoices section."""
        invoices = data.get("tenant_invoices", [])
        if not invoices:
            return "## 5. Tenant Invoices\n\n_No tenant invoices generated._"
        lines = [
            "## 5. Tenant Invoices\n",
            "| Invoice # | Tenant | Period | Amount | Due Date | Status |",
            "|-----------|--------|--------|--------|----------|--------|",
        ]
        for inv in invoices:
            lines.append(
                f"| {inv.get('invoice_number', '-')} "
                f"| {inv.get('tenant', '-')} "
                f"| {inv.get('period', '-')} "
                f"| {self._fmt_currency(inv.get('amount', 0))} "
                f"| {inv.get('due_date', '-')} "
                f"| {inv.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_variance(self, data: Dict[str, Any]) -> str:
        """Render variance analysis section."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return "## 6. Variance Analysis\n\n_No variance data available._"
        lines = [
            "## 6. Variance Analysis\n",
            "| Entity | Budget | Actual | Variance | Variance (%) | Explanation |",
            "|--------|--------|--------|----------|-------------|-------------|",
        ]
        for v in variances:
            var_val = v.get("variance", 0)
            marker = " !!!" if abs(v.get("variance_pct", 0)) > 10 else ""
            lines.append(
                f"| {v.get('entity', '-')} "
                f"| {self._fmt_currency(v.get('budget', 0))} "
                f"| {self._fmt_currency(v.get('actual', 0))} "
                f"| {self._fmt_currency(var_val)} "
                f"| {self._fmt(v.get('variance_pct', 0))}%{marker} "
                f"| {v.get('explanation', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Cost allocation based on configured methodology. "
            "Variances exceeding 10% are flagged for review.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("allocation_summary", {})
        return (
            f'<h1>Cost Allocation Report</h1>\n'
            f'<p class="subtitle">Property: {data.get("property_name", "-")} | '
            f'Period: {data.get("billing_period", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">{self._fmt_currency(summary.get("total_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Entities</span>'
            f'<span class="value">{summary.get("entity_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Meters</span>'
            f'<span class="value">{summary.get("meter_count", 0)}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Allocated</span>'
            f'<span class="value">100%</span></div>\n'
            f'</div>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology section."""
        meth = data.get("methodology", {})
        return (
            '<h2>Allocation Methodology</h2>\n'
            '<div class="info-box">'
            f'<p><strong>Primary:</strong> {meth.get("primary_method", "-")} | '
            f'<strong>Demand:</strong> {meth.get("demand_method", "-")} | '
            f'<strong>Shared:</strong> {meth.get("shared_services_method", "-")}</p></div>'
        )

    def _html_cost_by_entity(self, data: Dict[str, Any]) -> str:
        """Render HTML cost by entity table."""
        entities = data.get("cost_by_entity", [])
        rows = ""
        for e in entities:
            rows += (
                f'<tr><td>{e.get("name", "-")}</td>'
                f'<td>{e.get("type", "-")}</td>'
                f'<td>{self._fmt(e.get("consumption_kwh", 0), 0)}</td>'
                f'<td>{self._fmt_currency(e.get("energy_cost", 0))}</td>'
                f'<td>{self._fmt_currency(e.get("demand_cost", 0))}</td>'
                f'<td>{self._fmt_currency(e.get("shared_cost", 0))}</td>'
                f'<td>{self._fmt_currency(e.get("total_cost", 0))}</td>'
                f'<td>{self._fmt(e.get("cost_share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Cost by Entity</h2>\n'
            '<table>\n<tr><th>Entity</th><th>Type</th><th>kWh</th>'
            '<th>Energy</th><th>Demand</th><th>Shared</th>'
            f'<th>Total</th><th>Share</th></tr>\n{rows}</table>'
        )

    def _html_meter_mapping(self, data: Dict[str, Any]) -> str:
        """Render HTML meter-to-entity mapping."""
        mapping = data.get("meter_mapping", [])
        rows = ""
        for m in mapping:
            rows += (
                f'<tr><td>{m.get("meter_id", "-")}</td>'
                f'<td>{m.get("meter_name", "-")}</td>'
                f'<td>{m.get("entity", "-")}</td>'
                f'<td>{m.get("utility_type", "-")}</td>'
                f'<td>{m.get("allocation_method", "-")}</td></tr>\n'
            )
        return (
            '<h2>Meter-to-Entity Mapping</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Name</th><th>Entity</th>'
            f'<th>Utility</th><th>Method</th></tr>\n{rows}</table>'
        )

    def _html_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render HTML reconciliation section."""
        rec = data.get("reconciliation", {})
        status = rec.get("status", "Unreconciled")
        cls = "card-green" if status == "Reconciled" else "card-red"
        line_items = rec.get("line_items", [])
        rows = ""
        for li in line_items:
            rows += (
                f'<tr><td>{li.get("item", "-")}</td>'
                f'<td>{self._fmt_currency(li.get("billed", 0))}</td>'
                f'<td>{self._fmt_currency(li.get("allocated", 0))}</td>'
                f'<td>{self._fmt_currency(li.get("difference", 0))}</td></tr>\n'
            )
        return (
            '<h2>Reconciliation</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Billed</span>'
            f'<span class="value">{self._fmt_currency(rec.get("total_billed", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Total Allocated</span>'
            f'<span class="value">{self._fmt_currency(rec.get("total_allocated", 0))}</span></div>\n'
            f'  <div class="card {cls}"><span class="label">Status</span>'
            f'<span class="value">{status}</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Line Item</th><th>Billed</th>'
            f'<th>Allocated</th><th>Difference</th></tr>\n{rows}</table>'
        )

    def _html_tenant_invoices(self, data: Dict[str, Any]) -> str:
        """Render HTML tenant invoices section."""
        invoices = data.get("tenant_invoices", [])
        rows = ""
        for inv in invoices:
            rows += (
                f'<tr><td>{inv.get("invoice_number", "-")}</td>'
                f'<td>{inv.get("tenant", "-")}</td>'
                f'<td>{inv.get("period", "-")}</td>'
                f'<td>{self._fmt_currency(inv.get("amount", 0))}</td>'
                f'<td>{inv.get("due_date", "-")}</td>'
                f'<td>{inv.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Tenant Invoices</h2>\n'
            '<table>\n<tr><th>Invoice #</th><th>Tenant</th><th>Period</th>'
            f'<th>Amount</th><th>Due</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_variance(self, data: Dict[str, Any]) -> str:
        """Render HTML variance analysis section."""
        variances = data.get("variance_analysis", [])
        rows = ""
        for v in variances:
            var_pct = v.get("variance_pct", 0)
            cls = "variance-over" if abs(var_pct) > 10 else ""
            rows += (
                f'<tr class="{cls}"><td>{v.get("entity", "-")}</td>'
                f'<td>{self._fmt_currency(v.get("budget", 0))}</td>'
                f'<td>{self._fmt_currency(v.get("actual", 0))}</td>'
                f'<td>{self._fmt_currency(v.get("variance", 0))}</td>'
                f'<td>{self._fmt(var_pct)}%</td></tr>\n'
            )
        return (
            '<h2>Variance Analysis</h2>\n'
            '<table>\n<tr><th>Entity</th><th>Budget</th><th>Actual</th>'
            f'<th>Variance</th><th>Variance (%)</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        entities = data.get("cost_by_entity", [])
        variances = data.get("variance_analysis", [])
        return {
            "cost_by_entity_pie": {
                "type": "pie",
                "labels": [e.get("name", "") for e in entities],
                "values": [e.get("total_cost", 0) for e in entities],
            },
            "cost_breakdown_stacked_bar": {
                "type": "stacked_bar",
                "labels": [e.get("name", "") for e in entities],
                "series": {
                    "energy": [e.get("energy_cost", 0) for e in entities],
                    "demand": [e.get("demand_cost", 0) for e in entities],
                    "shared": [e.get("shared_cost", 0) for e in entities],
                },
            },
            "variance_bar": {
                "type": "bar",
                "labels": [v.get("entity", "") for v in variances],
                "series": {
                    "budget": [v.get("budget", 0) for v in variances],
                    "actual": [v.get("actual", 0) for v in variances],
                },
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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
            ".variance-over{background:#fff3cd !important;}"
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
