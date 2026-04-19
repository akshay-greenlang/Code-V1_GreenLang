# -*- coding: utf-8 -*-
"""
UtilitySavingsReportTemplate - Savings tracking and verification report for PACK-036.

Generates utility savings reports covering total savings by category,
implementation status of efficiency measures, verified vs projected
savings comparison (IPMVP-aligned), ROI achievement tracking, cumulative
savings trends, and measure-level performance detail. Designed for
energy managers and finance teams tracking utility cost reduction
programmes.

Sections:
    1. Header & Savings Summary
    2. Savings by Category
    3. Implementation Status
    4. Verified vs Projected
    5. ROI Analysis
    6. Cumulative Trend
    7. Measure Detail
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

class MeasureStatus(str, Enum):
    """Savings measure implementation status."""
    PLANNED = "Planned"
    IN_PROGRESS = "In Progress"
    COMPLETE = "Complete"
    VERIFIED = "Verified"
    CANCELLED = "Cancelled"

class UtilitySavingsReportTemplate:
    """
    Utility savings tracking and verification report template.

    Renders savings reports including category breakdowns, implementation
    status, IPMVP-aligned verification, ROI analysis, cumulative trends,
    and measure-level detail across markdown, HTML, JSON, and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UtilitySavingsReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render savings report as Markdown.

        Args:
            data: Savings tracking data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_savings_by_category(data),
            self._md_implementation_status(data),
            self._md_verified_vs_projected(data),
            self._md_roi_analysis(data),
            self._md_cumulative_trend(data),
            self._md_measure_detail(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render savings report as self-contained HTML.

        Args:
            data: Savings tracking data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_savings_by_category(data),
            self._html_implementation_status(data),
            self._html_verified_vs_projected(data),
            self._html_roi_analysis(data),
            self._html_cumulative_trend(data),
            self._html_measure_detail(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Utility Savings Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render savings report as structured JSON.

        Args:
            data: Savings tracking data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "utility_savings_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "savings_summary": data.get("savings_summary", {}),
            "savings_by_category": data.get("savings_by_category", []),
            "implementation_status": data.get("implementation_status", {}),
            "verified_vs_projected": data.get("verified_vs_projected", {}),
            "roi_analysis": data.get("roi_analysis", {}),
            "cumulative_trend": data.get("cumulative_trend", []),
            "measures": data.get("measures", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render savings measures as CSV.

        Args:
            data: Savings tracking data from engine processing.

        Returns:
            CSV string with one row per savings measure.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Measure ID", "Measure Name", "Category", "Status",
            "Projected Savings", "Verified Savings", "Realization Rate (%)",
            "Investment Cost", "ROI (%)", "Payback (months)",
            "Start Date", "Verification Date",
        ])
        for m in data.get("measures", []):
            writer.writerow([
                m.get("measure_id", ""),
                m.get("name", ""),
                m.get("category", ""),
                m.get("status", ""),
                self._fmt_raw(m.get("projected_savings", 0)),
                self._fmt_raw(m.get("verified_savings", 0)),
                self._fmt_raw(m.get("realization_rate_pct", 0)),
                self._fmt_raw(m.get("investment_cost", 0)),
                self._fmt_raw(m.get("roi_pct", 0)),
                self._fmt_raw(m.get("payback_months", 0), 0),
                m.get("start_date", ""),
                m.get("verification_date", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with savings summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("savings_summary", {})
        return (
            "# Utility Savings Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Total Projected Savings:** {self._fmt_currency(summary.get('total_projected', 0))}  \n"
            f"**Total Verified Savings:** {self._fmt_currency(summary.get('total_verified', 0))}  \n"
            f"**Realization Rate:** {self._fmt(summary.get('realization_rate_pct', 0))}%  \n"
            f"**Total Investment:** {self._fmt_currency(summary.get('total_investment', 0))}  \n"
            f"**Portfolio ROI:** {self._fmt(summary.get('portfolio_roi_pct', 0))}%  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 UtilitySavingsReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_savings_by_category(self, data: Dict[str, Any]) -> str:
        """Render savings by category section."""
        categories = data.get("savings_by_category", [])
        if not categories:
            return "## 1. Savings by Category\n\n_No savings data available._"
        lines = [
            "## 1. Savings by Category\n",
            "| Category | Projected | Verified | Realization (%) | Share (%) |",
            "|----------|----------|----------|----------------|----------|",
        ]
        for c in categories:
            lines.append(
                f"| {c.get('category', '-')} "
                f"| {self._fmt_currency(c.get('projected', 0))} "
                f"| {self._fmt_currency(c.get('verified', 0))} "
                f"| {self._fmt(c.get('realization_pct', 0))}% "
                f"| {self._fmt(c.get('share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_implementation_status(self, data: Dict[str, Any]) -> str:
        """Render implementation status section."""
        status = data.get("implementation_status", {})
        pipeline = status.get("pipeline", [])
        lines = [
            "## 2. Implementation Status\n",
            f"**Total Measures:** {status.get('total_measures', 0)}  ",
            f"**Planned:** {status.get('planned', 0)}  ",
            f"**In Progress:** {status.get('in_progress', 0)}  ",
            f"**Complete:** {status.get('complete', 0)}  ",
            f"**Verified:** {status.get('verified', 0)}  ",
            f"**Cancelled:** {status.get('cancelled', 0)}\n",
        ]
        if pipeline:
            lines.extend([
                "| Status | Count | Projected Savings | % of Total |",
                "|--------|-------|------------------|-----------|",
            ])
            for p in pipeline:
                lines.append(
                    f"| {p.get('status', '-')} "
                    f"| {p.get('count', 0)} "
                    f"| {self._fmt_currency(p.get('projected_savings', 0))} "
                    f"| {self._fmt(p.get('pct_of_total', 0))}% |"
                )
        return "\n".join(lines)

    def _md_verified_vs_projected(self, data: Dict[str, Any]) -> str:
        """Render verified vs projected comparison section."""
        vp = data.get("verified_vs_projected", {})
        comparisons = vp.get("comparisons", [])
        lines = [
            "## 3. Verified vs Projected Savings\n",
            f"**Verification Method:** {vp.get('verification_method', 'IPMVP Option C')}  ",
            f"**Baseline Period:** {vp.get('baseline_period', '-')}  ",
            f"**Reporting Period:** {vp.get('reporting_period', '-')}  ",
            f"**Overall Realization:** {self._fmt(vp.get('overall_realization_pct', 0))}%\n",
        ]
        if comparisons:
            lines.extend([
                "| Measure | Projected | Verified | Realization (%) | Status |",
                "|---------|----------|----------|----------------|--------|",
            ])
            for c in comparisons:
                real_pct = c.get("realization_pct", 0)
                marker = " !!!" if real_pct < 80 else ""
                lines.append(
                    f"| {c.get('measure', '-')} "
                    f"| {self._fmt_currency(c.get('projected', 0))} "
                    f"| {self._fmt_currency(c.get('verified', 0))} "
                    f"| {self._fmt(real_pct)}%{marker} "
                    f"| {c.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_roi_analysis(self, data: Dict[str, Any]) -> str:
        """Render ROI analysis section."""
        roi = data.get("roi_analysis", {})
        breakdown = roi.get("breakdown", [])
        lines = [
            "## 4. ROI Analysis\n",
            f"**Total Investment:** {self._fmt_currency(roi.get('total_investment', 0))}  ",
            f"**Annual Verified Savings:** {self._fmt_currency(roi.get('annual_verified_savings', 0))}  ",
            f"**Simple Payback:** {self._fmt(roi.get('simple_payback_years', 0))} years  ",
            f"**Portfolio ROI:** {self._fmt(roi.get('portfolio_roi_pct', 0))}%  ",
            f"**NPV (10-yr):** {self._fmt_currency(roi.get('npv_10yr', 0))}  ",
            f"**IRR:** {self._fmt(roi.get('irr_pct', 0))}%\n",
        ]
        if breakdown:
            lines.extend([
                "| Category | Investment | Annual Savings | Payback (yr) | ROI (%) |",
                "|----------|-----------|---------------|-------------|---------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('category', '-')} "
                    f"| {self._fmt_currency(b.get('investment', 0))} "
                    f"| {self._fmt_currency(b.get('annual_savings', 0))} "
                    f"| {self._fmt(b.get('payback_years', 0))} "
                    f"| {self._fmt(b.get('roi_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_cumulative_trend(self, data: Dict[str, Any]) -> str:
        """Render cumulative savings trend section."""
        trend = data.get("cumulative_trend", [])
        if not trend:
            return "## 5. Cumulative Savings Trend\n\n_No trend data available._"
        lines = [
            "## 5. Cumulative Savings Trend\n",
            "| Period | Period Savings | Cumulative Savings | Target | Achievement (%) |",
            "|--------|-------------|-------------------|--------|----------------|",
        ]
        for t in trend:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt_currency(t.get('period_savings', 0))} "
                f"| {self._fmt_currency(t.get('cumulative_savings', 0))} "
                f"| {self._fmt_currency(t.get('target', 0))} "
                f"| {self._fmt(t.get('achievement_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_measure_detail(self, data: Dict[str, Any]) -> str:
        """Render measure detail section."""
        measures = data.get("measures", [])
        if not measures:
            return "## 6. Measure Detail\n\n_No measures recorded._"
        lines = [
            "## 6. Measure Detail\n",
            "| ID | Measure | Category | Status | Projected | Verified | ROI (%) | Payback |",
            "|----|---------|----------|--------|----------|----------|---------|---------|",
        ]
        for m in measures:
            lines.append(
                f"| {m.get('measure_id', '-')} "
                f"| {m.get('name', '-')} "
                f"| {m.get('category', '-')} "
                f"| {m.get('status', '-')} "
                f"| {self._fmt_currency(m.get('projected_savings', 0))} "
                f"| {self._fmt_currency(m.get('verified_savings', 0))} "
                f"| {self._fmt(m.get('roi_pct', 0))}% "
                f"| {self._fmt(m.get('payback_months', 0), 0)}mo |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Savings verification follows IPMVP (International Performance "
            "Measurement and Verification Protocol). Measures with realization "
            "below 80% are flagged for investigation.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with savings summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("savings_summary", {})
        real_rate = summary.get("realization_rate_pct", 0)
        real_cls = "card-green" if real_rate >= 90 else (
            "card-red" if real_rate < 70 else ""
        )
        return (
            f'<h1>Utility Savings Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Period: {data.get("reporting_period", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Projected Savings</span>'
            f'<span class="value">{self._fmt_currency(summary.get("total_projected", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Verified Savings</span>'
            f'<span class="value">{self._fmt_currency(summary.get("total_verified", 0))}</span></div>\n'
            f'  <div class="card {real_cls}"><span class="label">Realization Rate</span>'
            f'<span class="value">{self._fmt(real_rate)}%</span></div>\n'
            f'  <div class="card"><span class="label">Portfolio ROI</span>'
            f'<span class="value">{self._fmt(summary.get("portfolio_roi_pct", 0))}%</span></div>\n'
            f'</div>'
        )

    def _html_savings_by_category(self, data: Dict[str, Any]) -> str:
        """Render HTML savings by category table."""
        categories = data.get("savings_by_category", [])
        rows = ""
        for c in categories:
            rows += (
                f'<tr><td>{c.get("category", "-")}</td>'
                f'<td>{self._fmt_currency(c.get("projected", 0))}</td>'
                f'<td>{self._fmt_currency(c.get("verified", 0))}</td>'
                f'<td>{self._fmt(c.get("realization_pct", 0))}%</td>'
                f'<td>{self._fmt(c.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Savings by Category</h2>\n'
            '<table>\n<tr><th>Category</th><th>Projected</th>'
            '<th>Verified</th><th>Realization</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_implementation_status(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation status section."""
        status = data.get("implementation_status", {})
        pipeline = status.get("pipeline", [])
        cards = (
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total</span>'
            f'<span class="value">{status.get("total_measures", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">In Progress</span>'
            f'<span class="value">{status.get("in_progress", 0)}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Verified</span>'
            f'<span class="value">{status.get("verified", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Cancelled</span>'
            f'<span class="value">{status.get("cancelled", 0)}</span></div>\n'
            f'</div>'
        )
        rows = ""
        for p in pipeline:
            rows += (
                f'<tr><td>{p.get("status", "-")}</td>'
                f'<td>{p.get("count", 0)}</td>'
                f'<td>{self._fmt_currency(p.get("projected_savings", 0))}</td>'
                f'<td>{self._fmt(p.get("pct_of_total", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Implementation Status</h2>\n'
            f'{cards}\n'
            '<table>\n<tr><th>Status</th><th>Count</th>'
            f'<th>Projected Savings</th><th>% of Total</th></tr>\n{rows}</table>'
        )

    def _html_verified_vs_projected(self, data: Dict[str, Any]) -> str:
        """Render HTML verified vs projected section."""
        vp = data.get("verified_vs_projected", {})
        comparisons = vp.get("comparisons", [])
        rows = ""
        for c in comparisons:
            real_pct = c.get("realization_pct", 0)
            cls = "underperform" if real_pct < 80 else ""
            rows += (
                f'<tr class="{cls}"><td>{c.get("measure", "-")}</td>'
                f'<td>{self._fmt_currency(c.get("projected", 0))}</td>'
                f'<td>{self._fmt_currency(c.get("verified", 0))}</td>'
                f'<td>{self._fmt(real_pct)}%</td>'
                f'<td>{c.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Verified vs Projected Savings</h2>\n'
            f'<div class="info-box"><p>Method: {vp.get("verification_method", "IPMVP")} | '
            f'Overall Realization: {self._fmt(vp.get("overall_realization_pct", 0))}%</p></div>\n'
            '<table>\n<tr><th>Measure</th><th>Projected</th>'
            '<th>Verified</th><th>Realization</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_roi_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML ROI analysis section."""
        roi = data.get("roi_analysis", {})
        breakdown = roi.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("category", "-")}</td>'
                f'<td>{self._fmt_currency(b.get("investment", 0))}</td>'
                f'<td>{self._fmt_currency(b.get("annual_savings", 0))}</td>'
                f'<td>{self._fmt(b.get("payback_years", 0))}</td>'
                f'<td>{self._fmt(b.get("roi_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>ROI Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Investment</span>'
            f'<span class="value">{self._fmt_currency(roi.get("total_investment", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._fmt_currency(roi.get("annual_verified_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(roi.get("simple_payback_years", 0))} yr</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(roi.get("irr_pct", 0))}%</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Category</th><th>Investment</th>'
            '<th>Annual Savings</th><th>Payback (yr)</th>'
            f'<th>ROI</th></tr>\n{rows}</table>'
        )

    def _html_cumulative_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative savings trend."""
        trend = data.get("cumulative_trend", [])
        rows = ""
        for t in trend:
            ach = t.get("achievement_pct", 0)
            cls = "card-green" if ach >= 100 else ("underperform" if ach < 80 else "")
            rows += (
                f'<tr class="{cls}"><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt_currency(t.get("period_savings", 0))}</td>'
                f'<td>{self._fmt_currency(t.get("cumulative_savings", 0))}</td>'
                f'<td>{self._fmt_currency(t.get("target", 0))}</td>'
                f'<td>{self._fmt(ach)}%</td></tr>\n'
            )
        return (
            '<h2>Cumulative Savings Trend</h2>\n'
            '<table>\n<tr><th>Period</th><th>Period Savings</th>'
            '<th>Cumulative</th><th>Target</th>'
            f'<th>Achievement</th></tr>\n{rows}</table>'
        )

    def _html_measure_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML measure detail section."""
        measures = data.get("measures", [])
        rows = ""
        for m in measures:
            status = m.get("status", "")
            cls = "measure-verified" if status == "Verified" else (
                "measure-cancelled" if status == "Cancelled" else ""
            )
            rows += (
                f'<tr class="{cls}"><td>{m.get("measure_id", "-")}</td>'
                f'<td>{m.get("name", "-")}</td>'
                f'<td>{m.get("category", "-")}</td>'
                f'<td>{status}</td>'
                f'<td>{self._fmt_currency(m.get("projected_savings", 0))}</td>'
                f'<td>{self._fmt_currency(m.get("verified_savings", 0))}</td>'
                f'<td>{self._fmt(m.get("roi_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Measure Detail</h2>\n'
            '<table>\n<tr><th>ID</th><th>Measure</th><th>Category</th>'
            '<th>Status</th><th>Projected</th><th>Verified</th>'
            f'<th>ROI</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        categories = data.get("savings_by_category", [])
        trend = data.get("cumulative_trend", [])
        comparisons = data.get("verified_vs_projected", {}).get("comparisons", [])
        return {
            "savings_by_category_pie": {
                "type": "pie",
                "labels": [c.get("category", "") for c in categories],
                "values": [c.get("verified", 0) for c in categories],
            },
            "cumulative_trend_line": {
                "type": "line",
                "labels": [t.get("period", "") for t in trend],
                "series": {
                    "cumulative": [t.get("cumulative_savings", 0) for t in trend],
                    "target": [t.get("target", 0) for t in trend],
                },
            },
            "verified_vs_projected_bar": {
                "type": "bar",
                "labels": [c.get("measure", "") for c in comparisons],
                "series": {
                    "projected": [c.get("projected", 0) for c in comparisons],
                    "verified": [c.get("verified", 0) for c in comparisons],
                },
            },
            "realization_gauge": {
                "type": "gauge",
                "value": data.get("savings_summary", {}).get(
                    "realization_rate_pct", 0
                ),
                "target": 100,
                "thresholds": {"red": 70, "amber": 90, "green": 100},
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
            ".underperform{background:#f8d7da !important;}"
            ".measure-verified{background:#d1e7dd !important;}"
            ".measure-cancelled{background:#e9ecef !important;color:#6c757d;}"
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
