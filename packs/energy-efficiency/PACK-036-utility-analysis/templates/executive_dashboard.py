# -*- coding: utf-8 -*-
"""
ExecutiveDashboardTemplate - C-suite utility KPI dashboard for PACK-036.

Generates executive-level utility dashboards with key KPI cards, cost
trend charts, budget RAG (Red/Amber/Green) status, anomaly alerts,
savings achieved to date, and prioritized action items. Designed for
board-level presentation with concise, high-impact visualizations.

Sections:
    1. Header & Executive Summary
    2. KPI Cards
    3. Cost Trend
    4. Budget RAG Status
    5. Anomaly Alerts
    6. Savings Summary
    7. Action Items
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


class RAGStatus(str, Enum):
    """Budget RAG status levels."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class ExecutiveDashboardTemplate:
    """
    C-suite executive utility KPI dashboard template.

    Renders executive dashboards with KPI cards, cost trend data,
    budget RAG status indicators, anomaly alerts, savings summaries,
    and prioritized action items across markdown, HTML, JSON, and
    CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RAG_DESCRIPTIONS: Dict[str, str] = {
        "GREEN": "On track - within 5% of budget",
        "AMBER": "Warning - 5-10% above budget",
        "RED": "Critical - more than 10% above budget",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive dashboard as Markdown.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_cards(data),
            self._md_cost_trend(data),
            self._md_budget_rag(data),
            self._md_anomalies(data),
            self._md_savings_summary(data),
            self._md_action_items(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive dashboard as self-contained HTML.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_cards(data),
            self._html_cost_trend(data),
            self._html_budget_rag(data),
            self._html_anomalies(data),
            self._html_savings_summary(data),
            self._html_action_items(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Utility Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive dashboard as structured JSON.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Dict with structured dashboard sections and provenance hash.
        """
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "executive_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "kpis": data.get("kpis", {}),
            "cost_trend": data.get("cost_trend", []),
            "budget_rag": data.get("budget_rag", {}),
            "anomalies": data.get("anomalies", []),
            "savings_summary": data.get("savings_summary", {}),
            "action_items": data.get("action_items", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render action items as CSV.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            CSV string with one row per action item.
        """
        self.generated_at = _utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Priority", "Action", "Owner", "Due Date",
            "Status", "Impact", "Category",
        ])
        for item in data.get("action_items", []):
            writer.writerow([
                item.get("priority", ""),
                item.get("action", ""),
                item.get("owner", ""),
                item.get("due_date", ""),
                item.get("status", ""),
                self._fmt_raw(item.get("impact", 0)),
                item.get("category", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with executive summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("executive_summary", {})
        return (
            "# Executive Utility Dashboard\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Total Utility Spend (YTD):** {self._fmt_currency(summary.get('ytd_spend', 0))}  \n"
            f"**Budget Status:** {summary.get('budget_status', '-')}  \n"
            f"**Savings Achieved:** {self._fmt_currency(summary.get('savings_achieved', 0))}  \n"
            f"**Open Action Items:** {summary.get('open_actions', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 ExecutiveDashboardTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render KPI cards section."""
        kpis = data.get("kpis", {})
        cards = kpis.get("cards", [])
        if not cards:
            return "## 1. Key Performance Indicators\n\n_No KPI data available._"
        lines = ["## 1. Key Performance Indicators\n"]
        for card in cards:
            trend_icon = "+" if card.get("trend_direction", "") == "up" else "-"
            lines.append(
                f"- **{card.get('label', '-')}:** {card.get('value', '-')} "
                f"({trend_icon}{self._fmt(abs(card.get('trend_pct', 0)))}% "
                f"{card.get('trend_period', 'YoY')})"
            )
        return "\n".join(lines)

    def _md_cost_trend(self, data: Dict[str, Any]) -> str:
        """Render cost trend section."""
        trend = data.get("cost_trend", [])
        if not trend:
            return "## 2. Cost Trend\n\n_No trend data available._"
        lines = [
            "## 2. Cost Trend\n",
            "| Month | Actual Cost | Budget | Variance | Variance (%) |",
            "|-------|-----------|--------|----------|-------------|",
        ]
        for t in trend:
            var_val = t.get("variance", 0)
            marker = " !!!" if abs(t.get("variance_pct", 0)) > 10 else ""
            lines.append(
                f"| {t.get('month', '-')} "
                f"| {self._fmt_currency(t.get('actual', 0))} "
                f"| {self._fmt_currency(t.get('budget', 0))} "
                f"| {self._fmt_currency(var_val)} "
                f"| {self._fmt(t.get('variance_pct', 0))}%{marker} |"
            )
        return "\n".join(lines)

    def _md_budget_rag(self, data: Dict[str, Any]) -> str:
        """Render budget RAG status section."""
        rag = data.get("budget_rag", {})
        accounts = rag.get("accounts", [])
        lines = [
            "## 3. Budget RAG Status\n",
            f"**Overall Status:** {rag.get('overall_status', '-')}  ",
            f"**YTD Budget:** {self._fmt_currency(rag.get('ytd_budget', 0))}  ",
            f"**YTD Actual:** {self._fmt_currency(rag.get('ytd_actual', 0))}  ",
            f"**YTD Variance:** {self._fmt_currency(rag.get('ytd_variance', 0))} "
            f"({self._fmt(rag.get('ytd_variance_pct', 0))}%)\n",
        ]
        if accounts:
            lines.extend([
                "| Account | Budget | Actual | Variance | Status |",
                "|---------|--------|--------|----------|--------|",
            ])
            for a in accounts:
                lines.append(
                    f"| {a.get('account', '-')} "
                    f"| {self._fmt_currency(a.get('budget', 0))} "
                    f"| {self._fmt_currency(a.get('actual', 0))} "
                    f"| {self._fmt_currency(a.get('variance', 0))} "
                    f"| {a.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_anomalies(self, data: Dict[str, Any]) -> str:
        """Render anomaly alerts section."""
        anomalies = data.get("anomalies", [])
        if not anomalies:
            return "## 4. Anomaly Alerts\n\n_No anomalies detected._"
        lines = [
            "## 4. Anomaly Alerts\n",
            "| # | Date | Account | Type | Severity | Description | Impact |",
            "|---|------|---------|------|----------|-------------|--------|",
        ]
        for i, a in enumerate(anomalies, 1):
            lines.append(
                f"| {i} | {a.get('date', '-')} "
                f"| {a.get('account', '-')} "
                f"| {a.get('type', '-')} "
                f"| {a.get('severity', '-')} "
                f"| {a.get('description', '-')} "
                f"| {self._fmt_currency(a.get('impact', 0))} |"
            )
        return "\n".join(lines)

    def _md_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render savings summary section."""
        ss = data.get("savings_summary", {})
        measures = ss.get("measures", [])
        lines = [
            "## 5. Savings Summary\n",
            f"**Total Savings (YTD):** {self._fmt_currency(ss.get('ytd_savings', 0))}  ",
            f"**Annual Target:** {self._fmt_currency(ss.get('annual_target', 0))}  ",
            f"**Achievement:** {self._fmt(ss.get('achievement_pct', 0))}%  ",
            f"**Measures Implemented:** {ss.get('measures_implemented', 0)}  ",
            f"**Measures In Pipeline:** {ss.get('measures_pipeline', 0)}\n",
        ]
        if measures:
            lines.extend([
                "| Measure | Category | Status | Annual Savings | Verified |",
                "|---------|----------|--------|---------------|----------|",
            ])
            for m in measures:
                lines.append(
                    f"| {m.get('measure', '-')} "
                    f"| {m.get('category', '-')} "
                    f"| {m.get('status', '-')} "
                    f"| {self._fmt_currency(m.get('annual_savings', 0))} "
                    f"| {m.get('verified', '-')} |"
                )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items section."""
        items = data.get("action_items", [])
        if not items:
            return "## 6. Action Items\n\n_No open action items._"
        lines = [
            "## 6. Action Items\n",
            "| # | Priority | Action | Owner | Due Date | Status |",
            "|---|----------|--------|-------|----------|--------|",
        ]
        for i, item in enumerate(items, 1):
            lines.append(
                f"| {i} | {item.get('priority', '-')} "
                f"| {item.get('action', '-')} "
                f"| {item.get('owner', '-')} "
                f"| {item.get('due_date', '-')} "
                f"| {item.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Dashboard data refreshed as of reporting period end date.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with executive summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("executive_summary", {})
        status = summary.get("budget_status", "GREEN")
        status_cls = (
            "card-green" if status == "GREEN" else
            ("card-amber" if status == "AMBER" else "card-red")
        )
        return (
            f'<h1>Executive Utility Dashboard</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Period: {data.get("reporting_period", "-")} | Generated: {ts}</p>'
        )

    def _html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards section."""
        kpis = data.get("kpis", {})
        cards = kpis.get("cards", [])
        card_html = ""
        for card in cards:
            trend_dir = card.get("trend_direction", "flat")
            trend_cls = "trend-up" if trend_dir == "up" else (
                "trend-down" if trend_dir == "down" else ""
            )
            trend_icon = "+" if trend_dir == "up" else "-" if trend_dir == "down" else ""
            card_html += (
                f'  <div class="card"><span class="label">{card.get("label", "-")}</span>'
                f'<span class="value">{card.get("value", "-")}</span>'
                f'<span class="label {trend_cls}">{trend_icon}'
                f'{self._fmt(abs(card.get("trend_pct", 0)))}% '
                f'{card.get("trend_period", "YoY")}</span></div>\n'
            )
        return f'<h2>Key Performance Indicators</h2>\n<div class="summary-cards">\n{card_html}</div>'

    def _html_cost_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML cost trend table."""
        trend = data.get("cost_trend", [])
        rows = ""
        for t in trend:
            var_pct = t.get("variance_pct", 0)
            cls = "variance-over" if abs(var_pct) > 10 else ""
            rows += (
                f'<tr class="{cls}"><td>{t.get("month", "-")}</td>'
                f'<td>{self._fmt_currency(t.get("actual", 0))}</td>'
                f'<td>{self._fmt_currency(t.get("budget", 0))}</td>'
                f'<td>{self._fmt_currency(t.get("variance", 0))}</td>'
                f'<td>{self._fmt(var_pct)}%</td></tr>\n'
            )
        return (
            '<h2>Cost Trend</h2>\n'
            '<table>\n<tr><th>Month</th><th>Actual</th><th>Budget</th>'
            f'<th>Variance</th><th>Variance (%)</th></tr>\n{rows}</table>'
        )

    def _html_budget_rag(self, data: Dict[str, Any]) -> str:
        """Render HTML budget RAG status section."""
        rag = data.get("budget_rag", {})
        status = rag.get("overall_status", "GREEN")
        rag_cls = (
            "rag-green" if status == "GREEN" else
            ("rag-amber" if status == "AMBER" else "rag-red")
        )
        accounts = rag.get("accounts", [])
        rows = ""
        for a in accounts:
            a_status = a.get("status", "GREEN")
            a_cls = (
                "rag-green" if a_status == "GREEN" else
                ("rag-amber" if a_status == "AMBER" else "rag-red")
            )
            rows += (
                f'<tr><td>{a.get("account", "-")}</td>'
                f'<td>{self._fmt_currency(a.get("budget", 0))}</td>'
                f'<td>{self._fmt_currency(a.get("actual", 0))}</td>'
                f'<td>{self._fmt_currency(a.get("variance", 0))}</td>'
                f'<td class="{a_cls}">{a_status}</td></tr>\n'
            )
        return (
            '<h2>Budget RAG Status</h2>\n'
            f'<div class="rag-indicator {rag_cls}"><strong>Overall: {status}</strong> - '
            f'{self.RAG_DESCRIPTIONS.get(status, "")}</div>\n'
            '<table>\n<tr><th>Account</th><th>Budget</th><th>Actual</th>'
            f'<th>Variance</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_anomalies(self, data: Dict[str, Any]) -> str:
        """Render HTML anomaly alerts section."""
        anomalies = data.get("anomalies", [])
        rows = ""
        for a in anomalies:
            sev = a.get("severity", "").lower()
            cls = "severity-critical" if sev == "critical" else (
                "severity-high" if sev == "high" else ""
            )
            rows += (
                f'<tr class="{cls}"><td>{a.get("date", "-")}</td>'
                f'<td>{a.get("account", "-")}</td>'
                f'<td>{a.get("type", "-")}</td>'
                f'<td>{a.get("severity", "-")}</td>'
                f'<td>{a.get("description", "-")}</td>'
                f'<td>{self._fmt_currency(a.get("impact", 0))}</td></tr>\n'
            )
        return (
            '<h2>Anomaly Alerts</h2>\n'
            '<table>\n<tr><th>Date</th><th>Account</th><th>Type</th>'
            '<th>Severity</th><th>Description</th>'
            f'<th>Impact</th></tr>\n{rows}</table>'
        )

    def _html_savings_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML savings summary section."""
        ss = data.get("savings_summary", {})
        achievement = ss.get("achievement_pct", 0)
        bar_color = "#198754" if achievement >= 80 else (
            "#ffc107" if achievement >= 50 else "#dc3545"
        )
        return (
            '<h2>Savings Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">YTD Savings</span>'
            f'<span class="value">{self._fmt_currency(ss.get("ytd_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Target</span>'
            f'<span class="value">{self._fmt_currency(ss.get("annual_target", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Achievement</span>'
            f'<span class="value">{self._fmt(achievement)}%</span></div>\n'
            f'  <div class="card"><span class="label">Measures</span>'
            f'<span class="value">{ss.get("measures_implemented", 0)} active</span></div>\n'
            '</div>\n'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(achievement, 100)}%;background:{bar_color};"></div></div>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items section."""
        items = data.get("action_items", [])
        rows = ""
        for item in items:
            priority = item.get("priority", "").lower()
            cls = "priority-high" if priority in ("critical", "high") else ""
            rows += (
                f'<tr class="{cls}"><td>{item.get("priority", "-")}</td>'
                f'<td>{item.get("action", "-")}</td>'
                f'<td>{item.get("owner", "-")}</td>'
                f'<td>{item.get("due_date", "-")}</td>'
                f'<td>{item.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Action Items</h2>\n'
            '<table>\n<tr><th>Priority</th><th>Action</th><th>Owner</th>'
            f'<th>Due Date</th><th>Status</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trend = data.get("cost_trend", [])
        kpi_cards = data.get("kpis", {}).get("cards", [])
        return {
            "cost_trend_line": {
                "type": "line",
                "labels": [t.get("month", "") for t in trend],
                "series": {
                    "actual": [t.get("actual", 0) for t in trend],
                    "budget": [t.get("budget", 0) for t in trend],
                },
            },
            "variance_bar": {
                "type": "bar",
                "labels": [t.get("month", "") for t in trend],
                "values": [t.get("variance", 0) for t in trend],
            },
            "kpi_summary": {
                "type": "kpi_cards",
                "cards": [
                    {
                        "label": c.get("label", ""),
                        "value": c.get("value", ""),
                        "trend": c.get("trend_pct", 0),
                    }
                    for c in kpi_cards
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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-amber{background:#fff3cd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
            ".rag-indicator{padding:12px 16px;border-radius:6px;margin:15px 0;font-size:1.1em;}"
            ".rag-green{background:#d1e7dd;color:#0f5132;}"
            ".rag-amber{background:#fff3cd;color:#664d03;}"
            ".rag-red{background:#f8d7da;color:#842029;}"
            ".trend-up{color:#dc3545;}"
            ".trend-down{color:#198754;}"
            ".variance-over{background:#fff3cd !important;}"
            ".severity-critical{background:#f8d7da !important;}"
            ".severity-high{background:#fff3cd !important;}"
            ".priority-high{background:#fff3cd !important;}"
            ".progress-bar{height:20px;background:#e9ecef;border-radius:10px;margin:15px 0;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:10px;}"
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
