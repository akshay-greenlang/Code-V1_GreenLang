# -*- coding: utf-8 -*-
"""
ProgressDashboardReportTemplate - Progress tracking dashboard for PACK-021.

Renders a progress tracking dashboard with KPI cards, year-over-year
emissions, actual vs pathway tracking, scope breakdown trends, intensity
metrics, action implementation status, gap analysis with RAG indicators,
corrective actions, and forecast projections.

Sections:
    1. Key Metrics Summary (KPI cards)
    2. Year-over-Year Emissions
    3. Progress vs Target (actual vs pathway)
    4. Scope Breakdown Trend
    5. Intensity Metrics Trend
    6. Action Implementation Status
    7. Gap Analysis (RAG status)
    8. Corrective Actions Required
    9. Forecast / Projection

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


def _pct_of(part: Any, total: Any) -> Decimal:
    p = Decimal(str(part))
    t = Decimal(str(total))
    if t == 0:
        return Decimal("0.00")
    return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _rag_status(actual: float, target: float) -> str:
    """Determine RAG status: on target, within 10%, or off track."""
    if target == 0:
        return "GREY"
    ratio = actual / target
    if ratio <= 1.0:
        return "GREEN"
    elif ratio <= 1.1:
        return "AMBER"
    return "RED"


class ProgressDashboardReportTemplate:
    """
    Progress tracking dashboard report template.

    Renders a comprehensive progress dashboard with KPIs, emissions
    tracking, gap analysis, corrective actions, and forecasts across
    markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_summary(data),
            self._md_yoy_emissions(data),
            self._md_progress_vs_target(data),
            self._md_scope_trend(data),
            self._md_intensity_trend(data),
            self._md_action_status(data),
            self._md_gap_analysis(data),
            self._md_corrective_actions(data),
            self._md_forecast(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_summary(data),
            self._html_yoy_emissions(data),
            self._html_progress_vs_target(data),
            self._html_scope_trend(data),
            self._html_intensity_trend(data),
            self._html_action_status(data),
            self._html_gap_analysis(data),
            self._html_corrective_actions(data),
            self._html_forecast(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Progress Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "progress_dashboard_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "kpis": data.get("kpis", {}),
            "yoy_emissions": data.get("yoy_emissions", []),
            "progress_vs_target": data.get("progress_vs_target", []),
            "scope_trend": data.get("scope_trend", []),
            "intensity_trend": data.get("intensity_trend", []),
            "action_status": data.get("action_status", []),
            "gap_analysis": data.get("gap_analysis", []),
            "corrective_actions": data.get("corrective_actions", []),
            "forecast": data.get("forecast", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Net Zero Progress Dashboard\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_kpi_summary(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", {})
        lines = [
            "## 1. Key Metrics Summary\n",
            "| KPI | Value | Unit | YoY Change | Status |",
            "|-----|------:|------|----------:|--------|",
        ]
        for kpi in kpis.get("items", []):
            change = kpi.get("yoy_change", 0)
            direction = "+" if change > 0 else ""
            lines.append(
                f"| {kpi.get('name', '-')} | {_dec_comma(kpi.get('value', 0))} "
                f"| {kpi.get('unit', '-')} "
                f"| {direction}{_dec(change)}% "
                f"| {kpi.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_yoy_emissions(self, data: Dict[str, Any]) -> str:
        yoy = data.get("yoy_emissions", [])
        lines = [
            "## 2. Year-over-Year Emissions\n",
            "| Year | Total (tCO2e) | Scope 1 | Scope 2 | Scope 3 | YoY Change (%) |",
            "|------|-------------:|--------:|--------:|--------:|---------------:|",
        ]
        prev_total = None
        for yr in yoy:
            total = Decimal(str(yr.get("total_tco2e", 0)))
            change_str = "-"
            if prev_total is not None and prev_total > 0:
                change = ((total - prev_total) / prev_total * Decimal("100"))
                change_str = f"{'+' if change > 0 else ''}{_dec(change)}%"
            prev_total = total
            lines.append(
                f"| {yr.get('year', '-')} | {_dec_comma(total)} "
                f"| {_dec_comma(yr.get('scope1_tco2e', 0))} "
                f"| {_dec_comma(yr.get('scope2_tco2e', 0))} "
                f"| {_dec_comma(yr.get('scope3_tco2e', 0))} "
                f"| {change_str} |"
            )
        return "\n".join(lines)

    def _md_progress_vs_target(self, data: Dict[str, Any]) -> str:
        pvt = data.get("progress_vs_target", [])
        lines = [
            "## 3. Progress vs Target\n",
            "| Year | Pathway Target (tCO2e) | Actual (tCO2e) | Variance | Status |",
            "|------|----------------------:|---------------:|---------:|--------|",
        ]
        for yr in pvt:
            target = Decimal(str(yr.get("target_tco2e", 0)))
            actual = Decimal(str(yr.get("actual_tco2e", 0)))
            variance = actual - target
            status = _rag_status(float(actual), float(target)) if target > 0 else "GREY"
            lines.append(
                f"| {yr.get('year', '-')} | {_dec_comma(target)} "
                f"| {_dec_comma(actual)} "
                f"| {'+' if variance > 0 else ''}{_dec_comma(variance)} "
                f"| {status} |"
            )
        return "\n".join(lines)

    def _md_scope_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("scope_trend", [])
        lines = [
            "## 4. Scope Breakdown Trend\n",
            "| Year | Scope 1 (tCO2e) | S1 Share | Scope 2 (tCO2e) | S2 Share | Scope 3 (tCO2e) | S3 Share |",
            "|------|----------------:|---------:|----------------:|---------:|----------------:|---------:|",
        ]
        for yr in trend:
            s1 = Decimal(str(yr.get("scope1_tco2e", 0)))
            s2 = Decimal(str(yr.get("scope2_tco2e", 0)))
            s3 = Decimal(str(yr.get("scope3_tco2e", 0)))
            total = s1 + s2 + s3
            lines.append(
                f"| {yr.get('year', '-')} "
                f"| {_dec_comma(s1)} | {_dec(_pct_of(s1, total))}% "
                f"| {_dec_comma(s2)} | {_dec(_pct_of(s2, total))}% "
                f"| {_dec_comma(s3)} | {_dec(_pct_of(s3, total))}% |"
            )
        return "\n".join(lines)

    def _md_intensity_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("intensity_trend", [])
        lines = [
            "## 5. Intensity Metrics Trend\n",
            "| Year | Revenue Intensity (tCO2e/EUR M) | Employee Intensity (tCO2e/FTE) | Area Intensity (tCO2e/sqm) |",
            "|------|-------------------------------:|-------------------------------:|---------------------------:|",
        ]
        for yr in trend:
            lines.append(
                f"| {yr.get('year', '-')} "
                f"| {_dec(yr.get('revenue_intensity', 0), 4)} "
                f"| {_dec(yr.get('employee_intensity', 0), 4)} "
                f"| {_dec(yr.get('area_intensity', 0), 4)} |"
            )
        return "\n".join(lines)

    def _md_action_status(self, data: Dict[str, Any]) -> str:
        actions = data.get("action_status", [])
        lines = [
            "## 6. Action Implementation Status\n",
            "| # | Action | Phase | Expected Abatement | Actual Abatement | Completion (%) | Status |",
            "|---|--------|-------|-------------------:|----------------:|---------------:|--------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('name', '-')} | {a.get('phase', '-')} "
                f"| {_dec_comma(a.get('expected_abatement_tco2e', 0))} "
                f"| {_dec_comma(a.get('actual_abatement_tco2e', 0))} "
                f"| {_dec(a.get('completion_pct', 0))}% "
                f"| {a.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        gaps = data.get("gap_analysis", [])
        lines = [
            "## 7. Gap Analysis\n",
            "| Area | Target (tCO2e) | Actual (tCO2e) | Gap (tCO2e) | RAG | Root Cause |",
            "|------|---------------:|---------------:|------------:|:---:|-----------|",
        ]
        for g in gaps:
            target = Decimal(str(g.get("target_tco2e", 0)))
            actual = Decimal(str(g.get("actual_tco2e", 0)))
            gap = actual - target
            rag = _rag_status(float(actual), float(target))
            lines.append(
                f"| {g.get('area', '-')} | {_dec_comma(target)} "
                f"| {_dec_comma(actual)} "
                f"| {'+' if gap > 0 else ''}{_dec_comma(gap)} "
                f"| {rag} | {g.get('root_cause', '-')} |"
            )
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        lines = ["## 8. Corrective Actions Required\n"]
        if actions:
            for i, a in enumerate(actions, 1):
                lines.append(
                    f"### {i}. {a.get('title', '')}\n"
                )
                lines.append(f"- **Area:** {a.get('area', '-')}")
                lines.append(f"- **Gap:** {_dec_comma(a.get('gap_tco2e', 0))} tCO2e")
                lines.append(f"- **Action:** {a.get('action', '-')}")
                lines.append(f"- **Owner:** {a.get('owner', '-')}")
                lines.append(f"- **Deadline:** {a.get('deadline', '-')}")
                lines.append("")
        else:
            lines.append("_No corrective actions required. All areas are on track._")
        return "\n".join(lines)

    def _md_forecast(self, data: Dict[str, Any]) -> str:
        forecast = data.get("forecast", [])
        lines = [
            "## 9. Forecast / Projection\n",
            "| Year | Projected Total (tCO2e) | Target (tCO2e) | On Track |",
            "|------|------------------------:|---------------:|:--------:|",
        ]
        for yr in forecast:
            projected = Decimal(str(yr.get("projected_tco2e", 0)))
            target = Decimal(str(yr.get("target_tco2e", 0)))
            on_track = "Yes" if projected <= target else "No"
            lines.append(
                f"| {yr.get('year', '-')} | {_dec_comma(projected)} "
                f"| {_dec_comma(target)} | {on_track} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*Progress dashboard with RAG gap analysis.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f5f7f5;color:#1a1a2e;}"
            ".report{max-width:1300px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".kpi-card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:20px;text-align:center;border-left:4px solid #2e7d32;}"
            ".kpi-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".kpi-value{font-size:1.6em;font-weight:700;color:#1b5e20;margin:4px 0;}"
            ".kpi-unit{font-size:0.75em;color:#689f38;}"
            ".kpi-change{font-size:0.85em;margin-top:4px;}"
            ".change-positive{color:#c62828;}"
            ".change-negative{color:#1b5e20;}"
            ".rag-green{background:#c8e6c9;color:#1b5e20;font-weight:600;padding:4px 10px;"
            "border-radius:12px;}"
            ".rag-amber{background:#fff9c4;color:#f57f17;font-weight:600;padding:4px 10px;"
            "border-radius:12px;}"
            ".rag-red{background:#ffcdd2;color:#c62828;font-weight:600;padding:4px 10px;"
            "border-radius:12px;}"
            ".rag-grey{background:#e0e0e0;color:#616161;font-weight:600;padding:4px 10px;"
            "border-radius:12px;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:16px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:#43a047;}"
            ".fill-amber{background:#ff8f00;}"
            ".fill-red{background:#e53935;}"
            ".corrective-card{border:1px solid #ffcdd2;border-left:4px solid #c62828;"
            "border-radius:8px;padding:16px;margin:12px 0;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Net Zero Progress Dashboard</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Year:</strong> {data.get("reporting_year", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_kpi_summary(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", {})
        cards = ""
        for kpi in kpis.get("items", []):
            change = kpi.get("yoy_change", 0)
            change_cls = "change-positive" if change > 0 else "change-negative"
            direction = "+" if change > 0 else ""
            cards += (
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{kpi.get("name", "-")}</div>'
                f'<div class="kpi-value">{_dec_comma(kpi.get("value", 0))}</div>'
                f'<div class="kpi-unit">{kpi.get("unit", "")}</div>'
                f'<div class="kpi-change {change_cls}">{direction}{_dec(change)}% YoY</div>'
                f'</div>\n'
            )
        return f'<h2>1. Key Metrics Summary</h2>\n<div class="kpi-grid">\n{cards}</div>'

    def _html_yoy_emissions(self, data: Dict[str, Any]) -> str:
        yoy = data.get("yoy_emissions", [])
        rows = ""
        prev_total = None
        for yr in yoy:
            total = Decimal(str(yr.get("total_tco2e", 0)))
            change_str = "-"
            if prev_total is not None and prev_total > 0:
                change = ((total - prev_total) / prev_total * Decimal("100"))
                change_str = f"{'+' if change > 0 else ''}{_dec(change)}%"
            prev_total = total
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(total)}</td>'
                f'<td>{_dec_comma(yr.get("scope1_tco2e", 0))}</td>'
                f'<td>{_dec_comma(yr.get("scope2_tco2e", 0))}</td>'
                f'<td>{_dec_comma(yr.get("scope3_tco2e", 0))}</td>'
                f'<td>{change_str}</td></tr>\n'
            )
        return (
            f'<h2>2. Year-over-Year Emissions</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Total (tCO2e)</th><th>Scope 1</th>'
            f'<th>Scope 2</th><th>Scope 3</th><th>YoY Change</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_progress_vs_target(self, data: Dict[str, Any]) -> str:
        pvt = data.get("progress_vs_target", [])
        rows = ""
        for yr in pvt:
            target = Decimal(str(yr.get("target_tco2e", 0)))
            actual = Decimal(str(yr.get("actual_tco2e", 0)))
            variance = actual - target
            rag = _rag_status(float(actual), float(target)) if target > 0 else "GREY"
            rag_cls = f"rag-{rag.lower()}"
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(target)}</td>'
                f'<td>{_dec_comma(actual)}</td>'
                f'<td>{"+" if variance > 0 else ""}{_dec_comma(variance)}</td>'
                f'<td><span class="{rag_cls}">{rag}</span></td></tr>\n'
            )
        return (
            f'<h2>3. Progress vs Target</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Pathway Target</th><th>Actual</th>'
            f'<th>Variance</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("scope_trend", [])
        rows = ""
        for yr in trend:
            s1 = Decimal(str(yr.get("scope1_tco2e", 0)))
            s2 = Decimal(str(yr.get("scope2_tco2e", 0)))
            s3 = Decimal(str(yr.get("scope3_tco2e", 0)))
            total = s1 + s2 + s3
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(s1)}</td><td>{_dec(_pct_of(s1, total))}%</td>'
                f'<td>{_dec_comma(s2)}</td><td>{_dec(_pct_of(s2, total))}%</td>'
                f'<td>{_dec_comma(s3)}</td><td>{_dec(_pct_of(s3, total))}%</td></tr>\n'
            )
        return (
            f'<h2>4. Scope Breakdown Trend</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Scope 1</th><th>S1 %</th>'
            f'<th>Scope 2</th><th>S2 %</th>'
            f'<th>Scope 3</th><th>S3 %</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_intensity_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("intensity_trend", [])
        rows = ""
        for yr in trend:
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec(yr.get("revenue_intensity", 0), 4)}</td>'
                f'<td>{_dec(yr.get("employee_intensity", 0), 4)}</td>'
                f'<td>{_dec(yr.get("area_intensity", 0), 4)}</td></tr>\n'
            )
        return (
            f'<h2>5. Intensity Metrics Trend</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Revenue (tCO2e/EUR M)</th>'
            f'<th>Employee (tCO2e/FTE)</th><th>Area (tCO2e/sqm)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_action_status(self, data: Dict[str, Any]) -> str:
        actions = data.get("action_status", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            completion = float(Decimal(str(a.get("completion_pct", 0))))
            bar_color = "fill-green" if completion >= 80 else "fill-amber" if completion >= 40 else "fill-red"
            rows += (
                f'<tr><td>{i}</td><td>{a.get("name", "-")}</td>'
                f'<td>{a.get("phase", "-")}</td>'
                f'<td>{_dec_comma(a.get("expected_abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(a.get("actual_abatement_tco2e", 0))}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill {bar_color}" '
                f'style="width:{completion}%"></div></div> {_dec(completion)}%</td>'
                f'<td>{a.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Action Implementation Status</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Phase</th><th>Expected</th>'
            f'<th>Actual</th><th>Completion</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        gaps = data.get("gap_analysis", [])
        rows = ""
        for g in gaps:
            target = Decimal(str(g.get("target_tco2e", 0)))
            actual = Decimal(str(g.get("actual_tco2e", 0)))
            gap = actual - target
            rag = _rag_status(float(actual), float(target))
            rag_cls = f"rag-{rag.lower()}"
            rows += (
                f'<tr><td>{g.get("area", "-")}</td>'
                f'<td>{_dec_comma(target)}</td>'
                f'<td>{_dec_comma(actual)}</td>'
                f'<td>{"+" if gap > 0 else ""}{_dec_comma(gap)}</td>'
                f'<td><span class="{rag_cls}">{rag}</span></td>'
                f'<td>{g.get("root_cause", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Gap Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>Area</th><th>Target</th><th>Actual</th>'
            f'<th>Gap</th><th>RAG</th><th>Root Cause</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        items = ""
        for i, a in enumerate(actions, 1):
            items += (
                f'<div class="corrective-card">'
                f'<strong>{i}. {a.get("title", "")}</strong><br>'
                f'<small>Area: {a.get("area", "-")} | '
                f'Gap: {_dec_comma(a.get("gap_tco2e", 0))} tCO2e | '
                f'Owner: {a.get("owner", "-")} | '
                f'Deadline: {a.get("deadline", "-")}</small>'
                f'<p>{a.get("action", "")}</p></div>\n'
            )
        return f'<h2>8. Corrective Actions</h2>\n{items}'

    def _html_forecast(self, data: Dict[str, Any]) -> str:
        forecast = data.get("forecast", [])
        rows = ""
        for yr in forecast:
            projected = Decimal(str(yr.get("projected_tco2e", 0)))
            target = Decimal(str(yr.get("target_tco2e", 0)))
            on_track = projected <= target
            cls = "rag-green" if on_track else "rag-red"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(projected)}</td>'
                f'<td>{_dec_comma(target)}</td>'
                f'<td><span class="{cls}">{label}</span></td></tr>\n'
            )
        return (
            f'<h2>9. Forecast / Projection</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Projected (tCO2e)</th><th>Target (tCO2e)</th>'
            f'<th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
