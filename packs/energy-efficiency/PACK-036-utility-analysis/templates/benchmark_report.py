# -*- coding: utf-8 -*-
"""
BenchmarkReportTemplate - Peer benchmarking report for PACK-036.

Generates utility benchmarking reports with EUI comparison against peer
buildings, cost-per-square-meter analysis, Energy Star score equivalents,
weather-normalized performance tracking, percentile rankings, and
improvement target recommendations. Designed for portfolio managers and
facility teams comparing building performance across peer groups.

Sections:
    1. Header & Benchmark Summary
    2. EUI Comparison
    3. Energy Star / Rating
    4. Peer Ranking
    5. Weather-Normalized Performance
    6. Cost Benchmarking
    7. Trend Analysis
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

class BenchmarkReportTemplate:
    """
    Peer benchmarking report template.

    Renders utility benchmark results including EUI comparisons, Energy
    Star scores, peer rankings, weather-normalized performance, cost
    benchmarking, and trend analysis across markdown, HTML, JSON, and
    CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RATING_LABELS: Dict[str, str] = {
        "A": "Excellent",
        "B": "Good",
        "C": "Average",
        "D": "Below Average",
        "E": "Poor",
        "F": "Critical",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BenchmarkReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render benchmark report as Markdown.

        Args:
            data: Benchmark data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_eui_comparison(data),
            self._md_energy_star(data),
            self._md_peer_ranking(data),
            self._md_weather_normalized(data),
            self._md_cost_benchmark(data),
            self._md_trends(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render benchmark report as self-contained HTML.

        Args:
            data: Benchmark data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_eui_comparison(data),
            self._html_energy_star(data),
            self._html_peer_ranking(data),
            self._html_weather_normalized(data),
            self._html_cost_benchmark(data),
            self._html_trends(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Benchmark Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render benchmark report as structured JSON.

        Args:
            data: Benchmark data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "benchmark_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "benchmark_summary": data.get("benchmark_summary", {}),
            "eui_comparison": data.get("eui_comparison", {}),
            "energy_star": data.get("energy_star", {}),
            "peer_ranking": data.get("peer_ranking", {}),
            "weather_normalized": data.get("weather_normalized", {}),
            "cost_benchmark": data.get("cost_benchmark", {}),
            "trends": data.get("trends", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render benchmark metrics as CSV.

        Args:
            data: Benchmark data from engine processing.

        Returns:
            CSV string with benchmark metrics per period.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Period", "Site EUI", "Source EUI", "Weather-Norm EUI",
            "Energy Star Score", "Peer Percentile",
            "Cost per sqm", "Peer Group", "Rank",
        ])
        for t in data.get("trends", []):
            writer.writerow([
                t.get("period", ""),
                self._fmt_raw(t.get("site_eui", 0)),
                self._fmt_raw(t.get("source_eui", 0)),
                self._fmt_raw(t.get("weather_norm_eui", 0)),
                self._fmt_raw(t.get("energy_star_score", 0), 0),
                self._fmt_raw(t.get("peer_percentile", 0), 0),
                self._fmt_raw(t.get("cost_per_sqm", 0)),
                t.get("peer_group", ""),
                t.get("rank", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with benchmark summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("benchmark_summary", {})
        return (
            "# Utility Benchmark Report\n\n"
            f"**Facility:** {data.get('facility_name', '-')}  \n"
            f"**Building Type:** {data.get('building_type', '-')}  \n"
            f"**Gross Floor Area:** {self._fmt(data.get('gross_floor_area_sqm', 0), 0)} m2  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Site EUI:** {self._fmt(summary.get('site_eui', 0))} kWh/m2/yr  \n"
            f"**Energy Star Score:** {summary.get('energy_star_score', '-')}  \n"
            f"**Peer Percentile:** {self._fmt(summary.get('peer_percentile', 0), 0)}th  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 BenchmarkReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_eui_comparison(self, data: Dict[str, Any]) -> str:
        """Render EUI comparison section."""
        eui = data.get("eui_comparison", {})
        metrics = eui.get("metrics", [])
        lines = [
            "## 1. EUI Comparison\n",
            f"**Building Type Median EUI:** {self._fmt(eui.get('median_eui', 0))} kWh/m2/yr  ",
            f"**Your Site EUI:** {self._fmt(eui.get('your_eui', 0))} kWh/m2/yr  ",
            f"**Performance vs Median:** {self._fmt(eui.get('vs_median_pct', 0))}%  ",
            f"**Status:** {eui.get('status', '-')}\n",
        ]
        if metrics:
            lines.extend([
                "| Metric | Your Value | Peer Median | Best-in-Class | Gap |",
                "|--------|-----------|------------|--------------|-----|",
            ])
            for m in metrics:
                lines.append(
                    f"| {m.get('metric', '-')} "
                    f"| {self._fmt(m.get('value', 0))} "
                    f"| {self._fmt(m.get('peer_median', 0))} "
                    f"| {self._fmt(m.get('best_in_class', 0))} "
                    f"| {self._fmt(m.get('gap', 0))} |"
                )
        return "\n".join(lines)

    def _md_energy_star(self, data: Dict[str, Any]) -> str:
        """Render Energy Star / rating section."""
        es = data.get("energy_star", {})
        return (
            "## 2. Energy Star / Performance Rating\n\n"
            f"**Energy Star Score:** {es.get('score', '-')} / 100  \n"
            f"**Rating:** {es.get('rating', '-')} - "
            f"{self.RATING_LABELS.get(es.get('rating', ''), '-')}  \n"
            f"**Certification Eligible:** {es.get('certification_eligible', '-')}  \n"
            f"**Score vs Previous Year:** {self._fmt(es.get('yoy_change', 0))} points  \n"
            f"**National Percentile:** {self._fmt(es.get('national_percentile', 0), 0)}th  \n"
            f"**Points to Certification (75):** {es.get('points_to_cert', '-')}"
        )

    def _md_peer_ranking(self, data: Dict[str, Any]) -> str:
        """Render peer ranking section."""
        pr = data.get("peer_ranking", {})
        peers = pr.get("peer_list", [])
        lines = [
            "## 3. Peer Ranking\n",
            f"**Peer Group:** {pr.get('peer_group_name', '-')}  ",
            f"**Peer Count:** {pr.get('peer_count', 0)}  ",
            f"**Your Rank:** {pr.get('your_rank', '-')} / {pr.get('peer_count', 0)}  ",
            f"**Percentile:** {self._fmt(pr.get('percentile', 0), 0)}th  ",
            f"**Quartile:** Q{pr.get('quartile', '-')}\n",
        ]
        if peers:
            lines.extend([
                "| Rank | Building | EUI (kWh/m2) | Score | Status |",
                "|------|----------|-------------|-------|--------|",
            ])
            for p in peers:
                marker = " <-- YOU" if p.get("is_self", False) else ""
                lines.append(
                    f"| {p.get('rank', '-')} "
                    f"| {p.get('name', '-')} "
                    f"| {self._fmt(p.get('eui', 0))} "
                    f"| {p.get('score', '-')} "
                    f"| {p.get('status', '-')}{marker} |"
                )
        return "\n".join(lines)

    def _md_weather_normalized(self, data: Dict[str, Any]) -> str:
        """Render weather-normalized performance section."""
        wn = data.get("weather_normalized", {})
        comparison = wn.get("comparison", [])
        lines = [
            "## 4. Weather-Normalized Performance\n",
            f"**Normalized EUI:** {self._fmt(wn.get('normalized_eui', 0))} kWh/m2/yr  ",
            f"**Actual EUI:** {self._fmt(wn.get('actual_eui', 0))} kWh/m2/yr  ",
            f"**Weather Adjustment:** {self._fmt(wn.get('adjustment_pct', 0))}%  ",
            f"**HDD (Actual/Normal):** {self._fmt(wn.get('hdd_actual', 0), 0)} / "
            f"{self._fmt(wn.get('hdd_normal', 0), 0)}  ",
            f"**CDD (Actual/Normal):** {self._fmt(wn.get('cdd_actual', 0), 0)} / "
            f"{self._fmt(wn.get('cdd_normal', 0), 0)}\n",
        ]
        if comparison:
            lines.extend([
                "| Metric | Actual | Normalized | Peer Median |",
                "|--------|--------|-----------|------------|",
            ])
            for c in comparison:
                lines.append(
                    f"| {c.get('metric', '-')} "
                    f"| {self._fmt(c.get('actual', 0))} "
                    f"| {self._fmt(c.get('normalized', 0))} "
                    f"| {self._fmt(c.get('peer_median', 0))} |"
                )
        return "\n".join(lines)

    def _md_cost_benchmark(self, data: Dict[str, Any]) -> str:
        """Render cost benchmarking section."""
        cb = data.get("cost_benchmark", {})
        breakdown = cb.get("breakdown", [])
        lines = [
            "## 5. Cost Benchmarking\n",
            f"**Cost per m2:** {self._fmt_currency(cb.get('cost_per_sqm', 0))} /m2/yr  ",
            f"**Peer Median Cost:** {self._fmt_currency(cb.get('peer_median_cost_sqm', 0))} /m2/yr  ",
            f"**Cost Percentile:** {self._fmt(cb.get('cost_percentile', 0), 0)}th  ",
            f"**Blended Rate:** {self._fmt(cb.get('blended_rate', 0), 4)} /kWh  ",
            f"**Peer Median Rate:** {self._fmt(cb.get('peer_median_rate', 0), 4)} /kWh\n",
        ]
        if breakdown:
            lines.extend([
                "| Cost Category | Your Cost | Peer Median | Difference |",
                "|--------------|----------|------------|-----------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('category', '-')} "
                    f"| {self._fmt_currency(b.get('your_cost', 0))} "
                    f"| {self._fmt_currency(b.get('peer_median', 0))} "
                    f"| {self._fmt_currency(b.get('difference', 0))} |"
                )
        return "\n".join(lines)

    def _md_trends(self, data: Dict[str, Any]) -> str:
        """Render trend analysis section."""
        trends = data.get("trends", [])
        if not trends:
            return "## 6. Trend Analysis\n\n_No trend data available._"
        lines = [
            "## 6. Trend Analysis\n",
            "| Period | Site EUI | Source EUI | Score | Percentile | Cost/m2 |",
            "|--------|---------|----------|-------|-----------|---------|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt(t.get('site_eui', 0))} "
                f"| {self._fmt(t.get('source_eui', 0))} "
                f"| {t.get('energy_star_score', '-')} "
                f"| {self._fmt(t.get('peer_percentile', 0), 0)}th "
                f"| {self._fmt_currency(t.get('cost_per_sqm', 0))} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Benchmark comparisons use CBECS/CIBSE TM46 reference data. "
            "Weather normalization applied via degree-day regression.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with benchmark summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("benchmark_summary", {})
        score = summary.get("energy_star_score", 0)
        score_cls = "card-green" if score >= 75 else ("card-red" if score < 50 else "")
        return (
            f'<h1>Utility Benchmark Report</h1>\n'
            f'<p class="subtitle">Facility: {data.get("facility_name", "-")} | '
            f'Type: {data.get("building_type", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Site EUI</span>'
            f'<span class="value">{self._fmt(summary.get("site_eui", 0))} kWh/m2</span></div>\n'
            f'  <div class="card {score_cls}"><span class="label">Energy Star</span>'
            f'<span class="value">{score}</span></div>\n'
            f'  <div class="card"><span class="label">Peer Percentile</span>'
            f'<span class="value">{self._fmt(summary.get("peer_percentile", 0), 0)}th</span></div>\n'
            f'  <div class="card"><span class="label">Cost/m2</span>'
            f'<span class="value">{self._fmt_currency(summary.get("cost_per_sqm", 0))}</span></div>\n'
            f'</div>'
        )

    def _html_eui_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML EUI comparison section."""
        eui = data.get("eui_comparison", {})
        metrics = eui.get("metrics", [])
        rows = ""
        for m in metrics:
            rows += (
                f'<tr><td>{m.get("metric", "-")}</td>'
                f'<td>{self._fmt(m.get("value", 0))}</td>'
                f'<td>{self._fmt(m.get("peer_median", 0))}</td>'
                f'<td>{self._fmt(m.get("best_in_class", 0))}</td>'
                f'<td>{self._fmt(m.get("gap", 0))}</td></tr>\n'
            )
        return (
            '<h2>EUI Comparison</h2>\n'
            f'<div class="info-box"><p>Your EUI: {self._fmt(eui.get("your_eui", 0))} | '
            f'Median: {self._fmt(eui.get("median_eui", 0))} | '
            f'Performance: {self._fmt(eui.get("vs_median_pct", 0))}% vs median</p></div>\n'
            '<table>\n<tr><th>Metric</th><th>Your Value</th><th>Peer Median</th>'
            f'<th>Best-in-Class</th><th>Gap</th></tr>\n{rows}</table>'
        )

    def _html_energy_star(self, data: Dict[str, Any]) -> str:
        """Render HTML Energy Star section."""
        es = data.get("energy_star", {})
        score = es.get("score", 0)
        bar_pct = min(score, 100)
        bar_color = "#198754" if score >= 75 else ("#ffc107" if score >= 50 else "#dc3545")
        return (
            '<h2>Energy Star / Performance Rating</h2>\n'
            f'<div class="score-bar-container">'
            f'<div class="score-bar" style="width:{bar_pct}%;background:{bar_color};"></div>'
            f'<span class="score-label">{score} / 100</span></div>\n'
            f'<p>Rating: <strong>{es.get("rating", "-")}</strong> | '
            f'Certification Eligible: {es.get("certification_eligible", "-")} | '
            f'YoY Change: {self._fmt(es.get("yoy_change", 0))} pts</p>'
        )

    def _html_peer_ranking(self, data: Dict[str, Any]) -> str:
        """Render HTML peer ranking section."""
        pr = data.get("peer_ranking", {})
        peers = pr.get("peer_list", [])
        rows = ""
        for p in peers:
            cls = ' class="highlight-row"' if p.get("is_self", False) else ""
            rows += (
                f'<tr{cls}><td>{p.get("rank", "-")}</td>'
                f'<td>{p.get("name", "-")}</td>'
                f'<td>{self._fmt(p.get("eui", 0))}</td>'
                f'<td>{p.get("score", "-")}</td>'
                f'<td>{p.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Peer Ranking</h2>\n'
            f'<div class="info-box"><p>Peer Group: {pr.get("peer_group_name", "-")} | '
            f'Your Rank: {pr.get("your_rank", "-")} / {pr.get("peer_count", 0)} | '
            f'Quartile: Q{pr.get("quartile", "-")}</p></div>\n'
            '<table>\n<tr><th>Rank</th><th>Building</th><th>EUI</th>'
            f'<th>Score</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_weather_normalized(self, data: Dict[str, Any]) -> str:
        """Render HTML weather-normalized section."""
        wn = data.get("weather_normalized", {})
        comparison = wn.get("comparison", [])
        rows = ""
        for c in comparison:
            rows += (
                f'<tr><td>{c.get("metric", "-")}</td>'
                f'<td>{self._fmt(c.get("actual", 0))}</td>'
                f'<td>{self._fmt(c.get("normalized", 0))}</td>'
                f'<td>{self._fmt(c.get("peer_median", 0))}</td></tr>\n'
            )
        return (
            '<h2>Weather-Normalized Performance</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Actual</th>'
            f'<th>Normalized</th><th>Peer Median</th></tr>\n{rows}</table>'
        )

    def _html_cost_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML cost benchmarking section."""
        cb = data.get("cost_benchmark", {})
        breakdown = cb.get("breakdown", [])
        rows = ""
        for b in breakdown:
            diff = b.get("difference", 0)
            cls = "variance-over" if diff > 0 else ""
            rows += (
                f'<tr class="{cls}"><td>{b.get("category", "-")}</td>'
                f'<td>{self._fmt_currency(b.get("your_cost", 0))}</td>'
                f'<td>{self._fmt_currency(b.get("peer_median", 0))}</td>'
                f'<td>{self._fmt_currency(diff)}</td></tr>\n'
            )
        return (
            '<h2>Cost Benchmarking</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Your Cost/m2</span>'
            f'<span class="value">{self._fmt_currency(cb.get("cost_per_sqm", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Peer Median</span>'
            f'<span class="value">{self._fmt_currency(cb.get("peer_median_cost_sqm", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Cost Percentile</span>'
            f'<span class="value">{self._fmt(cb.get("cost_percentile", 0), 0)}th</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Category</th><th>Your Cost</th>'
            f'<th>Peer Median</th><th>Difference</th></tr>\n{rows}</table>'
        )

    def _html_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML trend analysis section."""
        trends = data.get("trends", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt(t.get("site_eui", 0))}</td>'
                f'<td>{t.get("energy_star_score", "-")}</td>'
                f'<td>{self._fmt(t.get("peer_percentile", 0), 0)}th</td>'
                f'<td>{self._fmt_currency(t.get("cost_per_sqm", 0))}</td></tr>\n'
            )
        return (
            '<h2>Trend Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Site EUI</th><th>Score</th>'
            f'<th>Percentile</th><th>Cost/m2</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trends = data.get("trends", [])
        peers = data.get("peer_ranking", {}).get("peer_list", [])
        return {
            "eui_trend_line": {
                "type": "line",
                "labels": [t.get("period", "") for t in trends],
                "series": {
                    "site_eui": [t.get("site_eui", 0) for t in trends],
                    "source_eui": [t.get("source_eui", 0) for t in trends],
                    "weather_norm": [t.get("weather_norm_eui", 0) for t in trends],
                },
            },
            "score_trend_line": {
                "type": "line",
                "labels": [t.get("period", "") for t in trends],
                "values": [t.get("energy_star_score", 0) for t in trends],
            },
            "peer_distribution_bar": {
                "type": "bar",
                "labels": [p.get("name", "") for p in peers],
                "values": [p.get("eui", 0) for p in peers],
                "highlight": next(
                    (p.get("name", "") for p in peers if p.get("is_self")), ""
                ),
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
            ".highlight-row{background:#fff3cd !important;}"
            ".variance-over{color:#dc3545;}"
            ".score-bar-container{position:relative;height:30px;background:#e9ecef;"
            "border-radius:6px;margin:15px 0;overflow:hidden;}"
            ".score-bar{height:100%;border-radius:6px;}"
            ".score-label{position:absolute;top:5px;right:10px;font-weight:700;}"
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
