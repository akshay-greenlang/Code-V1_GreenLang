# -*- coding: utf-8 -*-
"""
ProcurementStrategyReportTemplate - Procurement strategy report for PACK-036.

Generates procurement strategy reports covering energy market overviews,
contract comparison matrices, risk assessment profiles, green procurement
options (RECs, PPAs, green tariffs), procurement calendar with key dates,
and hedging strategy evaluations. Designed for energy procurement
professionals and CFOs managing utility supply contracts.

Sections:
    1. Header & Strategy Summary
    2. Market Overview
    3. Contract Comparison
    4. Risk Assessment
    5. Green Procurement Options
    6. Hedging Strategies
    7. Procurement Calendar
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

class ProcurementStrategyReportTemplate:
    """
    Procurement strategy report template.

    Renders procurement analysis including market overview, contract
    comparisons, risk profiles, green procurement options, hedging
    strategies, and procurement calendar across markdown, HTML, JSON,
    and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RISK_LABELS: Dict[str, str] = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Critical",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProcurementStrategyReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render procurement strategy report as Markdown.

        Args:
            data: Procurement strategy data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_market_overview(data),
            self._md_contract_comparison(data),
            self._md_risk_assessment(data),
            self._md_green_options(data),
            self._md_hedging(data),
            self._md_calendar(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render procurement strategy report as self-contained HTML.

        Args:
            data: Procurement strategy data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_market_overview(data),
            self._html_contract_comparison(data),
            self._html_risk_assessment(data),
            self._html_green_options(data),
            self._html_hedging(data),
            self._html_calendar(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Procurement Strategy Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render procurement strategy report as structured JSON.

        Args:
            data: Procurement strategy data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "procurement_strategy_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "strategy_summary": data.get("strategy_summary", {}),
            "market_overview": data.get("market_overview", {}),
            "contracts": data.get("contracts", []),
            "risk_assessment": data.get("risk_assessment", {}),
            "green_options": data.get("green_options", []),
            "hedging_strategies": data.get("hedging_strategies", []),
            "calendar": data.get("calendar", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render contract comparison as CSV.

        Args:
            data: Procurement strategy data from engine processing.

        Returns:
            CSV string with one row per contract option.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Supplier", "Contract Type", "Term (months)", "Price (per kWh)",
            "Annual Cost", "Green (%)", "Risk Level",
            "Flexibility", "Rating",
        ])
        for c in data.get("contracts", []):
            writer.writerow([
                c.get("supplier", ""),
                c.get("contract_type", ""),
                c.get("term_months", ""),
                self._fmt_raw(c.get("price_per_kwh", 0), 4),
                self._fmt_raw(c.get("annual_cost", 0)),
                self._fmt_raw(c.get("green_pct", 0)),
                c.get("risk_level", ""),
                c.get("flexibility", ""),
                c.get("rating", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with strategy summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("strategy_summary", {})
        return (
            "# Procurement Strategy Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Procurement Period:** {data.get('procurement_period', '-')}  \n"
            f"**Annual Load:** {self._fmt(summary.get('annual_load_mwh', 0), 0)} MWh  \n"
            f"**Current Spend:** {self._fmt_currency(summary.get('current_annual_spend', 0))}  \n"
            f"**Contracts Expiring:** {summary.get('contracts_expiring', 0)}  \n"
            f"**Recommended Strategy:** {summary.get('recommended_strategy', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 ProcurementStrategyReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_market_overview(self, data: Dict[str, Any]) -> str:
        """Render market overview section."""
        mkt = data.get("market_overview", {})
        forecasts = mkt.get("price_forecasts", [])
        lines = [
            "## 1. Market Overview\n",
            f"**Market Region:** {mkt.get('region', '-')}  ",
            f"**Current Spot Price:** {self._fmt(mkt.get('spot_price', 0), 4)} /kWh  ",
            f"**12-Month Forward:** {self._fmt(mkt.get('forward_12m', 0), 4)} /kWh  ",
            f"**YoY Price Change:** {self._fmt(mkt.get('yoy_change_pct', 0))}%  ",
            f"**Market Trend:** {mkt.get('trend', '-')}  ",
            f"**Volatility Index:** {self._fmt(mkt.get('volatility_index', 0))}  ",
            f"**Supply Outlook:** {mkt.get('supply_outlook', '-')}\n",
        ]
        if forecasts:
            lines.extend([
                "### Price Forecasts\n",
                "| Period | Low | Mid | High |",
                "|--------|-----|-----|------|",
            ])
            for f in forecasts:
                lines.append(
                    f"| {f.get('period', '-')} "
                    f"| {self._fmt(f.get('low', 0), 4)} "
                    f"| {self._fmt(f.get('mid', 0), 4)} "
                    f"| {self._fmt(f.get('high', 0), 4)} |"
                )
        return "\n".join(lines)

    def _md_contract_comparison(self, data: Dict[str, Any]) -> str:
        """Render contract comparison section."""
        contracts = data.get("contracts", [])
        if not contracts:
            return "## 2. Contract Comparison\n\n_No contracts to compare._"
        lines = [
            "## 2. Contract Comparison\n",
            "| Supplier | Type | Term | Price/kWh | Annual Cost | Green | Risk | Rating |",
            "|----------|------|------|----------|------------|-------|------|--------|",
        ]
        for c in contracts:
            lines.append(
                f"| {c.get('supplier', '-')} "
                f"| {c.get('contract_type', '-')} "
                f"| {c.get('term_months', '-')}mo "
                f"| {self._fmt(c.get('price_per_kwh', 0), 4)} "
                f"| {self._fmt_currency(c.get('annual_cost', 0))} "
                f"| {self._fmt(c.get('green_pct', 0), 0)}% "
                f"| {c.get('risk_level', '-')} "
                f"| {c.get('rating', '-')} |"
            )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        ra = data.get("risk_assessment", {})
        risks = ra.get("risks", [])
        lines = [
            "## 3. Risk Assessment\n",
            f"**Overall Risk Level:** {ra.get('overall_risk', '-')}  ",
            f"**Market Risk Exposure:** {self._fmt_currency(ra.get('market_exposure', 0))}  ",
            f"**Value at Risk (95%):** {self._fmt_currency(ra.get('var_95', 0))}\n",
        ]
        if risks:
            lines.extend([
                "| Risk Factor | Likelihood | Impact | Mitigation |",
                "|-------------|-----------|--------|-----------|",
            ])
            for r in risks:
                lines.append(
                    f"| {r.get('factor', '-')} "
                    f"| {r.get('likelihood', '-')} "
                    f"| {r.get('impact', '-')} "
                    f"| {r.get('mitigation', '-')} |"
                )
        return "\n".join(lines)

    def _md_green_options(self, data: Dict[str, Any]) -> str:
        """Render green procurement options section."""
        options = data.get("green_options", [])
        if not options:
            return "## 4. Green Procurement Options\n\n_No green options available._"
        lines = [
            "## 4. Green Procurement Options\n",
            "| Option | Type | Premium | Coverage (%) | Annual Cost | CO2 Avoided |",
            "|--------|------|---------|-------------|------------|------------|",
        ]
        for o in options:
            lines.append(
                f"| {o.get('name', '-')} "
                f"| {o.get('type', '-')} "
                f"| {self._fmt(o.get('premium_per_kwh', 0), 4)} /kWh "
                f"| {self._fmt(o.get('coverage_pct', 0), 0)}% "
                f"| {self._fmt_currency(o.get('annual_cost', 0))} "
                f"| {self._fmt(o.get('co2_avoided_tonnes', 0), 0)} tCO2 |"
            )
        return "\n".join(lines)

    def _md_hedging(self, data: Dict[str, Any]) -> str:
        """Render hedging strategies section."""
        strategies = data.get("hedging_strategies", [])
        if not strategies:
            return "## 5. Hedging Strategies\n\n_No hedging strategies evaluated._"
        lines = [
            "## 5. Hedging Strategies\n",
            "| Strategy | Coverage (%) | Cost | Max Upside | Max Downside | Recommendation |",
            "|----------|-------------|------|-----------|-------------|---------------|",
        ]
        for s in strategies:
            lines.append(
                f"| {s.get('strategy', '-')} "
                f"| {self._fmt(s.get('coverage_pct', 0), 0)}% "
                f"| {self._fmt_currency(s.get('cost', 0))} "
                f"| {self._fmt_currency(s.get('max_upside', 0))} "
                f"| {self._fmt_currency(s.get('max_downside', 0))} "
                f"| {s.get('recommendation', '-')} |"
            )
        return "\n".join(lines)

    def _md_calendar(self, data: Dict[str, Any]) -> str:
        """Render procurement calendar section."""
        events = data.get("calendar", [])
        if not events:
            return "## 6. Procurement Calendar\n\n_No calendar events._"
        lines = [
            "## 6. Procurement Calendar\n",
            "| Date | Event | Action Required | Status |",
            "|------|-------|----------------|--------|",
        ]
        for e in events:
            lines.append(
                f"| {e.get('date', '-')} "
                f"| {e.get('event', '-')} "
                f"| {e.get('action', '-')} "
                f"| {e.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Market data and price forecasts are indicative. "
            "Engage qualified energy broker for binding quotes.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with strategy summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("strategy_summary", {})
        return (
            f'<h1>Procurement Strategy Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Period: {data.get("procurement_period", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Load</span>'
            f'<span class="value">{self._fmt(summary.get("annual_load_mwh", 0), 0)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Current Spend</span>'
            f'<span class="value">{self._fmt_currency(summary.get("current_annual_spend", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Expiring Contracts</span>'
            f'<span class="value">{summary.get("contracts_expiring", 0)}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Strategy</span>'
            f'<span class="value">{summary.get("recommended_strategy", "-")}</span></div>\n'
            f'</div>'
        )

    def _html_market_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML market overview section."""
        mkt = data.get("market_overview", {})
        trend = mkt.get("trend", "-")
        trend_cls = "card-green" if "down" in trend.lower() else (
            "card-red" if "up" in trend.lower() else ""
        )
        return (
            '<h2>Market Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Spot Price</span>'
            f'<span class="value">{self._fmt(mkt.get("spot_price", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">12M Forward</span>'
            f'<span class="value">{self._fmt(mkt.get("forward_12m", 0), 4)}</span></div>\n'
            f'  <div class="card {trend_cls}"><span class="label">Trend</span>'
            f'<span class="value">{trend}</span></div>\n'
            f'  <div class="card"><span class="label">Volatility</span>'
            f'<span class="value">{self._fmt(mkt.get("volatility_index", 0))}</span></div>\n'
            '</div>'
        )

    def _html_contract_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML contract comparison table."""
        contracts = data.get("contracts", [])
        rows = ""
        for c in contracts:
            rows += (
                f'<tr><td>{c.get("supplier", "-")}</td>'
                f'<td>{c.get("contract_type", "-")}</td>'
                f'<td>{c.get("term_months", "-")}mo</td>'
                f'<td>{self._fmt(c.get("price_per_kwh", 0), 4)}</td>'
                f'<td>{self._fmt_currency(c.get("annual_cost", 0))}</td>'
                f'<td>{self._fmt(c.get("green_pct", 0), 0)}%</td>'
                f'<td>{c.get("risk_level", "-")}</td>'
                f'<td>{c.get("rating", "-")}</td></tr>\n'
            )
        return (
            '<h2>Contract Comparison</h2>\n'
            '<table>\n<tr><th>Supplier</th><th>Type</th><th>Term</th>'
            '<th>Price/kWh</th><th>Annual Cost</th><th>Green</th>'
            f'<th>Risk</th><th>Rating</th></tr>\n{rows}</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment section."""
        ra = data.get("risk_assessment", {})
        risks = ra.get("risks", [])
        rows = ""
        for r in risks:
            rows += (
                f'<tr><td>{r.get("factor", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td>{r.get("impact", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        overall = ra.get("overall_risk", "-").lower()
        risk_cls = "card-red" if overall in ("high", "critical") else (
            "card-green" if overall == "low" else ""
        )
        return (
            '<h2>Risk Assessment</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card {risk_cls}"><span class="label">Overall Risk</span>'
            f'<span class="value">{ra.get("overall_risk", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Market Exposure</span>'
            f'<span class="value">{self._fmt_currency(ra.get("market_exposure", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">VaR (95%)</span>'
            f'<span class="value">{self._fmt_currency(ra.get("var_95", 0))}</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Risk Factor</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_green_options(self, data: Dict[str, Any]) -> str:
        """Render HTML green procurement options."""
        options = data.get("green_options", [])
        rows = ""
        for o in options:
            rows += (
                f'<tr><td>{o.get("name", "-")}</td>'
                f'<td>{o.get("type", "-")}</td>'
                f'<td>{self._fmt(o.get("premium_per_kwh", 0), 4)}</td>'
                f'<td>{self._fmt(o.get("coverage_pct", 0), 0)}%</td>'
                f'<td>{self._fmt_currency(o.get("annual_cost", 0))}</td>'
                f'<td>{self._fmt(o.get("co2_avoided_tonnes", 0), 0)} tCO2</td></tr>\n'
            )
        return (
            '<h2>Green Procurement Options</h2>\n'
            '<table>\n<tr><th>Option</th><th>Type</th><th>Premium</th>'
            '<th>Coverage</th><th>Annual Cost</th>'
            f'<th>CO2 Avoided</th></tr>\n{rows}</table>'
        )

    def _html_hedging(self, data: Dict[str, Any]) -> str:
        """Render HTML hedging strategies section."""
        strategies = data.get("hedging_strategies", [])
        rows = ""
        for s in strategies:
            rows += (
                f'<tr><td>{s.get("strategy", "-")}</td>'
                f'<td>{self._fmt(s.get("coverage_pct", 0), 0)}%</td>'
                f'<td>{self._fmt_currency(s.get("cost", 0))}</td>'
                f'<td>{self._fmt_currency(s.get("max_upside", 0))}</td>'
                f'<td>{self._fmt_currency(s.get("max_downside", 0))}</td>'
                f'<td>{s.get("recommendation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Hedging Strategies</h2>\n'
            '<table>\n<tr><th>Strategy</th><th>Coverage</th><th>Cost</th>'
            '<th>Max Upside</th><th>Max Downside</th>'
            f'<th>Recommendation</th></tr>\n{rows}</table>'
        )

    def _html_calendar(self, data: Dict[str, Any]) -> str:
        """Render HTML procurement calendar."""
        events = data.get("calendar", [])
        rows = ""
        for e in events:
            rows += (
                f'<tr><td>{e.get("date", "-")}</td>'
                f'<td>{e.get("event", "-")}</td>'
                f'<td>{e.get("action", "-")}</td>'
                f'<td>{e.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Procurement Calendar</h2>\n'
            '<table>\n<tr><th>Date</th><th>Event</th>'
            f'<th>Action Required</th><th>Status</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        contracts = data.get("contracts", [])
        forecasts = data.get("market_overview", {}).get("price_forecasts", [])
        green = data.get("green_options", [])
        return {
            "contract_cost_bar": {
                "type": "bar",
                "labels": [c.get("supplier", "") for c in contracts],
                "values": [c.get("annual_cost", 0) for c in contracts],
            },
            "price_forecast_line": {
                "type": "line",
                "labels": [f.get("period", "") for f in forecasts],
                "series": {
                    "low": [f.get("low", 0) for f in forecasts],
                    "mid": [f.get("mid", 0) for f in forecasts],
                    "high": [f.get("high", 0) for f in forecasts],
                },
            },
            "green_options_bar": {
                "type": "bar",
                "labels": [g.get("name", "") for g in green],
                "series": {
                    "cost": [g.get("annual_cost", 0) for g in green],
                    "co2_avoided": [g.get("co2_avoided_tonnes", 0) for g in green],
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
