# -*- coding: utf-8 -*-
"""
DisclosureDashboardTemplate - Race to Zero interactive dashboard for PACK-025.

Renders an interactive HTML dashboard with emissions trend visualization,
target pathway progress, credibility score gauge, partnership impact
metrics, verification status badges, and annual milestones timeline.

Sections:
    1. Dashboard Header & Summary Cards
    2. Emissions Trend Visualization (SVG chart)
    3. Target Pathway Progress (progress bars)
    4. Credibility Score Gauge (radial gauge)
    5. Scope Breakdown (donut-style breakdown)
    6. Partnership Impact Metrics
    7. Verification Status Badges
    8. Annual Milestones Timeline
    9. Key Actions & Achievements

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "disclosure_dashboard"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0


class DisclosureDashboardTemplate:
    """Race to Zero interactive disclosure dashboard template for PACK-025.

    Generates a rich interactive HTML dashboard with SVG-based charts,
    progress gauges, emissions trend lines, and verification badges.
    Also supports Markdown and JSON output modes.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the disclosure dashboard as Markdown (static version)."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary_cards(data),
            self._md_emissions_trend(data),
            self._md_target_progress(data),
            self._md_credibility_score(data),
            self._md_scope_breakdown(data),
            self._md_partnership_impact(data),
            self._md_verification_status(data),
            self._md_milestones(data),
            self._md_key_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the interactive disclosure dashboard as HTML with embedded SVG charts."""
        self.generated_at = _utcnow()
        css = self._dashboard_css()
        body = "\n".join([
            self._html_header(data),
            self._html_summary_cards(data),
            self._html_emissions_chart(data),
            self._html_target_progress(data),
            self._html_credibility_gauge(data),
            self._html_scope_breakdown(data),
            self._html_partnership_metrics(data),
            self._html_verification_badges(data),
            self._html_milestones_timeline(data),
            self._html_actions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>Race to Zero - Disclosure Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="dashboard">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the disclosure dashboard data as structured JSON."""
        self.generated_at = _utcnow()
        emissions_trend = data.get("emissions_trend", [])
        current = data.get("current_year", {})
        baseline = data.get("baseline", {})
        credibility = data.get("credibility_score", {})

        base_total = baseline.get("total_tco2e", 0)
        curr_total = current.get("total_tco2e", 0)
        from_baseline = _safe_div(base_total - curr_total, max(base_total, 1)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "summary": {
                "current_emissions_tco2e": curr_total,
                "baseline_emissions_tco2e": base_total,
                "reduction_from_baseline_pct": round(from_baseline, 1),
                "credibility_score": credibility.get("overall", 0),
                "verification_status": data.get("verification_status", ""),
            },
            "emissions_trend": emissions_trend,
            "scope_breakdown": {
                "scope1_tco2e": current.get("scope1_tco2e", 0),
                "scope2_tco2e": current.get("scope2_tco2e", 0),
                "scope3_tco2e": current.get("scope3_tco2e", 0),
            },
            "target_progress": data.get("target_progress", {}),
            "partnership_metrics": data.get("partnership_metrics", {}),
            "milestones": data.get("milestones", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = _utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Emissions Trend
        trend = data.get("emissions_trend", [])
        trend_rows: List[Dict[str, Any]] = []
        for t in trend:
            trend_rows.append({
                "Year": t.get("year", ""),
                "Scope 1 (tCO2e)": t.get("scope1_tco2e", 0),
                "Scope 2 (tCO2e)": t.get("scope2_tco2e", 0),
                "Scope 3 (tCO2e)": t.get("scope3_tco2e", 0),
                "Total (tCO2e)": t.get("total_tco2e", 0),
                "Target (tCO2e)": t.get("target_tco2e", ""),
            })
        sheets["Emissions Trend"] = trend_rows

        # Sheet 2: Dashboard Summary
        current = data.get("current_year", {})
        baseline = data.get("baseline", {})
        credibility = data.get("credibility_score", {})
        sheets["Dashboard Summary"] = [
            {"Metric": "Current Year Emissions", "Value": current.get("total_tco2e", 0), "Unit": "tCO2e"},
            {"Metric": "Baseline Emissions", "Value": baseline.get("total_tco2e", 0), "Unit": "tCO2e"},
            {"Metric": "Reduction from Baseline", "Value": round(_safe_div(
                baseline.get("total_tco2e", 0) - current.get("total_tco2e", 0),
                max(baseline.get("total_tco2e", 1), 1)) * 100, 1), "Unit": "%"},
            {"Metric": "Credibility Score", "Value": credibility.get("overall", 0), "Unit": "/100"},
            {"Metric": "Verification Status", "Value": data.get("verification_status", ""), "Unit": ""},
        ]

        # Sheet 3: Milestones
        milestones = data.get("milestones", [])
        ms_rows: List[Dict[str, Any]] = []
        for ms in milestones:
            ms_rows.append({
                "Year": ms.get("year", ""),
                "Milestone": ms.get("milestone", ""),
                "Status": ms.get("status", ""),
                "Impact": ms.get("impact", ""),
            })
        sheets["Milestones"] = ms_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Disclosure Dashboard\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_summary_cards(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        baseline = data.get("baseline", {})
        credibility = data.get("credibility_score", {})
        curr_total = current.get("total_tco2e", 0)
        base_total = baseline.get("total_tco2e", 0)
        reduction = _safe_div(base_total - curr_total, max(base_total, 1)) * 100

        return (
            f"## Summary\n\n"
            f"| Current Emissions | Reduction from Baseline | Credibility Score | Verification |\n"
            f"|:-----------------:|:-----------------------:|:-----------------:|:------------:|\n"
            f"| {_dec_comma(curr_total)} tCO2e | {_pct(reduction)} | "
            f"{_dec(credibility.get('overall', 0), 1)}/100 | "
            f"{data.get('verification_status', 'N/A')} |"
        )

    def _md_emissions_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("emissions_trend", [])
        lines = ["## Emissions Trend\n"]
        if trend:
            lines.extend([
                "| Year | S1 (tCO2e) | S2 (tCO2e) | S3 (tCO2e) | Total | Target |",
                "|:----:|-----------:|-----------:|-----------:|------:|-------:|",
            ])
            for t in trend:
                target = t.get("target_tco2e", "")
                target_str = _dec_comma(target) if target != "" else "--"
                lines.append(
                    f"| {t.get('year', '-')} "
                    f"| {_dec_comma(t.get('scope1_tco2e', 0))} "
                    f"| {_dec_comma(t.get('scope2_tco2e', 0))} "
                    f"| {_dec_comma(t.get('scope3_tco2e', 0))} "
                    f"| {_dec_comma(t.get('total_tco2e', 0))} "
                    f"| {target_str} |"
                )
        else:
            lines.append("_Emissions trend data not available._")
        return "\n".join(lines)

    def _md_target_progress(self, data: Dict[str, Any]) -> str:
        tp = data.get("target_progress", {})
        return (
            f"## Target Pathway Progress\n\n"
            f"| Target | Progress | Expected | Status |\n"
            f"|--------|:--------:|:--------:|:------:|\n"
            f"| Interim ({tp.get('interim_year', 2030)}) | {_pct(tp.get('interim_progress_pct', 0))} "
            f"| {_pct(tp.get('interim_expected_pct', 0))} "
            f"| {tp.get('interim_status', 'N/A')} |\n"
            f"| Net-Zero ({tp.get('netzero_year', 2050)}) | {_pct(tp.get('netzero_progress_pct', 0))} "
            f"| {_pct(tp.get('netzero_expected_pct', 0))} "
            f"| {tp.get('netzero_status', 'N/A')} |"
        )

    def _md_credibility_score(self, data: Dict[str, Any]) -> str:
        cs = data.get("credibility_score", {})
        dims = cs.get("dimensions", [])
        lines = [
            f"## Credibility Score: {_dec(cs.get('overall', 0), 1)}/100\n",
        ]
        if dims:
            lines.extend([
                "| Dimension | Score | Max |",
                "|-----------|:-----:|:---:|",
            ])
            for dim in dims:
                lines.append(
                    f"| {dim.get('name', '-')} "
                    f"| {_dec(dim.get('score', 0), 1)} "
                    f"| {dim.get('max_score', 100)} |"
                )
        return "\n".join(lines)

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        total = current.get("total_tco2e", 0)
        s1 = current.get("scope1_tco2e", 0)
        s2 = current.get("scope2_tco2e", 0)
        s3 = current.get("scope3_tco2e", 0)
        return (
            f"## Scope Breakdown\n\n"
            f"| Scope | Emissions (tCO2e) | Share |\n|-------|------------------:|:-----:|\n"
            f"| Scope 1 | {_dec_comma(s1)} | {_pct(_safe_div(s1, max(total, 1)) * 100)} |\n"
            f"| Scope 2 | {_dec_comma(s2)} | {_pct(_safe_div(s2, max(total, 1)) * 100)} |\n"
            f"| Scope 3 | {_dec_comma(s3)} | {_pct(_safe_div(s3, max(total, 1)) * 100)} |\n"
            f"| **Total** | **{_dec_comma(total)}** | **100%** |"
        )

    def _md_partnership_impact(self, data: Dict[str, Any]) -> str:
        pm = data.get("partnership_metrics", {})
        return (
            f"## Partnership Impact\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Partner Organizations | {pm.get('partner_count', 0)} |\n"
            f"| Combined Reduction | {_dec_comma(pm.get('combined_reduction_tco2e', 0))} tCO2e |\n"
            f"| Joint Projects | {pm.get('joint_projects', 0)} |\n"
            f"| Supplier Engagement | {_pct(pm.get('supplier_engagement_pct', 0))} |"
        )

    def _md_verification_status(self, data: Dict[str, Any]) -> str:
        v = data.get("verification", {})
        return (
            f"## Verification Status\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Status | {v.get('status', 'Pending')} |\n"
            f"| Level | {v.get('level', 'N/A')} |\n"
            f"| Verifier | {v.get('verifier', 'N/A')} |\n"
            f"| Valid Until | {v.get('valid_until', 'N/A')} |"
        )

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = ["## Annual Milestones\n"]
        if milestones:
            for ms in milestones:
                status = ms.get("status", "")
                marker = "[x]" if status == "Completed" else "[ ]"
                lines.append(f"- {marker} **{ms.get('year', '')}**: {ms.get('milestone', '')} ({status})")
        else:
            lines.append("_Milestones to be defined._")
        return "\n".join(lines)

    def _md_key_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("key_actions", [])
        lines = ["## Key Actions & Achievements\n"]
        if actions:
            for action in actions:
                lines.append(f"- **{action.get('action', '')}**: {action.get('impact', '')}")
        else:
            lines.append("_Key actions to be documented._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML Dashboard (interactive with embedded SVG)                       #
    # ------------------------------------------------------------------ #

    def _dashboard_css(self) -> str:
        return (
            "*{box-sizing:border-box;margin:0;padding:0;}"
            "body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f4f0;color:#1a1a2e;}"
            ".dashboard{max-width:1400px;margin:0 auto;padding:20px;}"
            ".dashboard-header{background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#fff;"
            "padding:30px;border-radius:12px;margin-bottom:20px;}"
            ".dashboard-header h1{font-size:1.8em;margin-bottom:8px;}"
            ".dashboard-header p{opacity:0.9;font-size:0.95em;}"
            ".grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0;}"
            ".grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0;}"
            ".grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:20px 0;}"
            "@media(max-width:900px){.grid-4,.grid-3,.grid-2{grid-template-columns:1fr;}}"
            ".card{background:#fff;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}"
            ".card-header{font-size:0.85em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;"
            "margin-bottom:8px;}"
            ".card-value{font-size:2em;font-weight:700;color:#1b5e20;}"
            ".card-sub{font-size:0.85em;color:#689f38;margin-top:4px;}"
            ".metric-card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-left:4px solid #2e7d32;}"
            ".section{background:#fff;border-radius:12px;padding:24px;margin:20px 0;"
            "box-shadow:0 2px 8px rgba(0,0,0,0.06);}"
            ".section h2{color:#2e7d32;font-size:1.2em;margin-bottom:16px;border-left:4px solid #43a047;"
            "padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;color:#1b5e20;font-weight:600;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".progress-bar{background:#e0e0e0;border-radius:10px;height:24px;overflow:hidden;"
            "margin:6px 0;position:relative;}"
            ".progress-fill{height:100%;border-radius:10px;transition:width 0.5s ease;}"
            ".progress-label{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
            "font-size:0.8em;font-weight:600;color:#1a1a2e;}"
            ".fill-green{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".fill-amber{background:linear-gradient(90deg,#ff9800,#ffb74d);}"
            ".fill-red{background:linear-gradient(90deg,#ef5350,#ef9a9a);}"
            ".badge{display:inline-block;padding:4px 12px;border-radius:16px;font-size:0.85em;"
            "font-weight:600;margin:4px;}"
            ".badge-verified{background:#c8e6c9;color:#1b5e20;border:1px solid #43a047;}"
            ".badge-pending{background:#fff3e0;color:#e65100;border:1px solid #ff9800;}"
            ".badge-not-verified{background:#ffcdd2;color:#c62828;border:1px solid #ef5350;}"
            ".gauge-container{text-align:center;padding:20px;}"
            ".gauge-circle{width:160px;height:160px;border-radius:50%;margin:0 auto;"
            "display:flex;align-items:center;justify-content:center;flex-direction:column;}"
            ".gauge-score{font-size:2.5em;font-weight:700;color:#1b5e20;}"
            ".gauge-label{font-size:0.85em;color:#558b2f;margin-top:4px;}"
            ".gauge-excellent{border:8px solid #2e7d32;background:#e8f5e9;}"
            ".gauge-good{border:8px solid #43a047;background:#e8f5e9;}"
            ".gauge-fair{border:8px solid #ff9800;background:#fff3e0;}"
            ".gauge-poor{border:8px solid #ef5350;background:#ffebee;}"
            ".timeline{position:relative;padding:20px 0;}"
            ".timeline-item{position:relative;padding:10px 0 10px 30px;border-left:3px solid #c8e6c9;}"
            ".timeline-item:last-child{border-left:3px solid transparent;}"
            ".timeline-dot{position:absolute;left:-8px;top:12px;width:13px;height:13px;"
            "border-radius:50%;border:2px solid #fff;}"
            ".dot-completed{background:#43a047;}"
            ".dot-active{background:#ff9800;}"
            ".dot-planned{background:#e0e0e0;}"
            ".timeline-year{font-weight:700;color:#1b5e20;font-size:0.95em;}"
            ".timeline-text{color:#555;font-size:0.9em;margin-top:2px;}"
            ".chart-container{padding:10px;}"
            ".footer{text-align:center;padding:20px;color:#689f38;font-size:0.85em;margin-top:30px;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="dashboard-header">\n'
            f'  <h1>Race to Zero -- Disclosure Dashboard</h1>\n'
            f'  <p>{org} | Reporting Year: {year} | Generated: {ts}</p>\n'
            f'</div>'
        )

    def _html_summary_cards(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        baseline = data.get("baseline", {})
        credibility = data.get("credibility_score", {})
        curr_total = current.get("total_tco2e", 0)
        base_total = baseline.get("total_tco2e", 0)
        reduction = _safe_div(base_total - curr_total, max(base_total, 1)) * 100
        v_status = data.get("verification_status", "Pending")

        return (
            f'<div class="grid-4">\n'
            f'  <div class="card metric-card"><div class="card-header">Current Emissions</div>'
            f'<div class="card-value">{_dec_comma(curr_total)}</div>'
            f'<div class="card-sub">tCO2e</div></div>\n'
            f'  <div class="card metric-card"><div class="card-header">Reduction from Baseline</div>'
            f'<div class="card-value">{_pct(reduction)}</div>'
            f'<div class="card-sub">since {baseline.get("year", "base year")}</div></div>\n'
            f'  <div class="card metric-card"><div class="card-header">Credibility Score</div>'
            f'<div class="card-value">{_dec(credibility.get("overall", 0), 1)}</div>'
            f'<div class="card-sub">out of 100</div></div>\n'
            f'  <div class="card metric-card"><div class="card-header">Verification</div>'
            f'<div class="card-value">{v_status}</div></div>\n'
            f'</div>'
        )

    def _html_emissions_chart(self, data: Dict[str, Any]) -> str:
        trend = data.get("emissions_trend", [])
        if not trend:
            return '<div class="section"><h2>Emissions Trend</h2><p>No trend data available.</p></div>'

        # SVG bar chart
        max_total = max(t.get("total_tco2e", 0) for t in trend) if trend else 1
        chart_width = 700
        chart_height = 300
        bar_width = min(60, (chart_width - 100) // max(len(trend), 1))
        gap = 10

        bars = ""
        labels = ""
        for i, t in enumerate(trend):
            total = t.get("total_tco2e", 0)
            bar_height = int(_safe_div(total, max_total) * (chart_height - 60))
            x = 60 + i * (bar_width + gap)
            y = chart_height - 30 - bar_height
            bars += (f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                     f'fill="#43a047" rx="4" opacity="0.85"/>\n')
            # Target line if available
            target = t.get("target_tco2e", "")
            if target != "":
                target_y = chart_height - 30 - int(_safe_div(float(target), max_total) * (chart_height - 60))
                bars += (f'<line x1="{x - 2}" y1="{target_y}" x2="{x + bar_width + 2}" y2="{target_y}" '
                         f'stroke="#ef5350" stroke-width="2" stroke-dasharray="4,2"/>\n')
            labels += (f'<text x="{x + bar_width // 2}" y="{chart_height - 10}" '
                       f'text-anchor="middle" font-size="11" fill="#555">{t.get("year", "")}</text>\n')
            labels += (f'<text x="{x + bar_width // 2}" y="{y - 5}" '
                       f'text-anchor="middle" font-size="10" fill="#1b5e20">{_dec_comma(total)}</text>\n')

        # Y-axis labels
        for j in range(5):
            val = int(max_total * j / 4)
            y_pos = chart_height - 30 - int((chart_height - 60) * j / 4)
            labels += (f'<text x="55" y="{y_pos + 4}" text-anchor="end" font-size="10" fill="#888">'
                       f'{_dec_comma(val)}</text>\n')
            labels += (f'<line x1="58" y1="{y_pos}" x2="{60 + len(trend) * (bar_width + gap)}" y2="{y_pos}" '
                       f'stroke="#e0e0e0" stroke-width="1"/>\n')

        svg = (f'<svg width="100%" viewBox="0 0 {chart_width} {chart_height}" '
               f'preserveAspectRatio="xMidYMid meet">\n{labels}{bars}\n'
               f'<text x="10" y="15" font-size="11" fill="#888" transform="rotate(-90 10 150)">'
               f'tCO2e</text>\n</svg>')

        return (
            f'<div class="section">\n'
            f'  <h2>Emissions Trend</h2>\n'
            f'  <div class="chart-container">{svg}</div>\n'
            f'  <p style="font-size:0.8em;color:#888;">Green bars = actual emissions. '
            f'Red dashed = target pathway.</p>\n'
            f'</div>'
        )

    def _html_target_progress(self, data: Dict[str, Any]) -> str:
        tp = data.get("target_progress", {})
        interim_pct = tp.get("interim_progress_pct", 0)
        netzero_pct = tp.get("netzero_progress_pct", 0)
        interim_fill = "fill-green" if interim_pct >= 80 else ("fill-amber" if interim_pct >= 50 else "fill-red")
        netzero_fill = "fill-green" if netzero_pct >= 80 else ("fill-amber" if netzero_pct >= 50 else "fill-red")

        return (
            f'<div class="section">\n'
            f'  <h2>Target Pathway Progress</h2>\n'
            f'  <p><strong>Interim Target ({tp.get("interim_year", 2030)})</strong></p>\n'
            f'  <div class="progress-bar">'
            f'<div class="progress-fill {interim_fill}" style="width:{min(interim_pct, 100)}%"></div>'
            f'<div class="progress-label">{_pct(interim_pct)}</div></div>\n'
            f'  <p><strong>Net-Zero Target ({tp.get("netzero_year", 2050)})</strong></p>\n'
            f'  <div class="progress-bar">'
            f'<div class="progress-fill {netzero_fill}" style="width:{min(netzero_pct, 100)}%"></div>'
            f'<div class="progress-label">{_pct(netzero_pct)}</div></div>\n'
            f'</div>'
        )

    def _html_credibility_gauge(self, data: Dict[str, Any]) -> str:
        cs = data.get("credibility_score", {})
        overall = cs.get("overall", 0)
        gauge_class = ("gauge-excellent" if overall >= 75 else
                       ("gauge-good" if overall >= 60 else
                        ("gauge-fair" if overall >= 40 else "gauge-poor")))
        dims = cs.get("dimensions", [])
        dim_bars = ""
        for dim in dims:
            score = dim.get("score", 0)
            fill = "fill-green" if score >= 70 else ("fill-amber" if score >= 40 else "fill-red")
            dim_bars += (
                f'<div style="margin:6px 0;"><span style="display:inline-block;width:120px;'
                f'font-size:0.85em;">{dim.get("name", "")}</span>'
                f'<div class="progress-bar" style="display:inline-block;width:calc(100% - 180px);'
                f'vertical-align:middle;">'
                f'<div class="progress-fill {fill}" style="width:{score}%"></div></div>'
                f' <span style="font-size:0.85em;font-weight:600;">{_dec(score, 0)}</span></div>\n'
            )

        return (
            f'<div class="section">\n'
            f'  <h2>Credibility Score</h2>\n'
            f'  <div class="grid-2">\n'
            f'    <div class="gauge-container">'
            f'<div class="gauge-circle {gauge_class}">'
            f'<div class="gauge-score">{_dec(overall, 0)}</div>'
            f'<div class="gauge-label">/ 100</div></div></div>\n'
            f'    <div>{dim_bars}</div>\n'
            f'  </div>\n'
            f'</div>'
        )

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        total = current.get("total_tco2e", 0)
        s1 = current.get("scope1_tco2e", 0)
        s2 = current.get("scope2_tco2e", 0)
        s3 = current.get("scope3_tco2e", 0)

        return (
            f'<div class="section">\n'
            f'  <h2>Scope Breakdown</h2>\n'
            f'  <div class="grid-3">\n'
            f'    <div class="card"><div class="card-header">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(s1)}</div>'
            f'<div class="card-sub">{_pct(_safe_div(s1, max(total, 1)) * 100)} of total</div></div>\n'
            f'    <div class="card"><div class="card-header">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(s2)}</div>'
            f'<div class="card-sub">{_pct(_safe_div(s2, max(total, 1)) * 100)} of total</div></div>\n'
            f'    <div class="card"><div class="card-header">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(s3)}</div>'
            f'<div class="card-sub">{_pct(_safe_div(s3, max(total, 1)) * 100)} of total</div></div>\n'
            f'  </div>\n'
            f'</div>'
        )

    def _html_partnership_metrics(self, data: Dict[str, Any]) -> str:
        pm = data.get("partnership_metrics", {})
        return (
            f'<div class="section">\n'
            f'  <h2>Partnership Impact</h2>\n'
            f'  <div class="grid-4">\n'
            f'    <div class="card"><div class="card-header">Partners</div>'
            f'<div class="card-value">{pm.get("partner_count", 0)}</div></div>\n'
            f'    <div class="card"><div class="card-header">Combined Reduction</div>'
            f'<div class="card-value">{_dec_comma(pm.get("combined_reduction_tco2e", 0))}</div>'
            f'<div class="card-sub">tCO2e</div></div>\n'
            f'    <div class="card"><div class="card-header">Joint Projects</div>'
            f'<div class="card-value">{pm.get("joint_projects", 0)}</div></div>\n'
            f'    <div class="card"><div class="card-header">Supplier Engagement</div>'
            f'<div class="card-value">{_pct(pm.get("supplier_engagement_pct", 0))}</div></div>\n'
            f'  </div>\n'
            f'</div>'
        )

    def _html_verification_badges(self, data: Dict[str, Any]) -> str:
        v = data.get("verification", {})
        status = v.get("status", "Pending")
        badge_class = ("badge-verified" if status == "Verified" else
                       ("badge-pending" if status == "Pending" else "badge-not-verified"))
        badges = data.get("verification_badges", [])
        badge_html = ""
        for b in badges:
            b_class = ("badge-verified" if b.get("status") == "Verified" else
                       ("badge-pending" if b.get("status") == "Pending" else "badge-not-verified"))
            badge_html += f'<span class="badge {b_class}">{b.get("label", "")}: {b.get("status", "")}</span>\n'

        if not badge_html:
            badge_html = (
                f'<span class="badge {badge_class}">GHG Inventory: {status}</span>\n'
                f'<span class="badge badge-pending">Progress Report: Pending</span>\n'
                f'<span class="badge badge-pending">Target Validation: Pending</span>\n'
            )

        return (
            f'<div class="section">\n'
            f'  <h2>Verification Status</h2>\n'
            f'  <div style="padding:10px;">{badge_html}</div>\n'
            f'  <table><tr><th>Field</th><th>Value</th></tr>\n'
            f'  <tr><td>Verifier</td><td>{v.get("verifier", "N/A")}</td></tr>\n'
            f'  <tr><td>Level</td><td>{v.get("level", "N/A")}</td></tr>\n'
            f'  <tr><td>Valid Until</td><td>{v.get("valid_until", "N/A")}</td></tr>\n'
            f'  </table>\n'
            f'</div>'
        )

    def _html_milestones_timeline(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        items = ""
        for ms in milestones:
            status = ms.get("status", "Planned")
            dot_class = ("dot-completed" if status == "Completed" else
                         ("dot-active" if status == "In Progress" else "dot-planned"))
            items += (
                f'<div class="timeline-item">\n'
                f'  <div class="timeline-dot {dot_class}"></div>\n'
                f'  <div class="timeline-year">{ms.get("year", "")}</div>\n'
                f'  <div class="timeline-text">{ms.get("milestone", "")} '
                f'<span class="badge badge-{"verified" if status == "Completed" else "pending"}">'
                f'{status}</span></div>\n'
                f'</div>\n'
            )
        if not items:
            items = '<p style="color:#888;"><em>Milestones to be defined.</em></p>'
        return (
            f'<div class="section">\n'
            f'  <h2>Annual Milestones</h2>\n'
            f'  <div class="timeline">{items}</div>\n'
            f'</div>'
        )

    def _html_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("key_actions", [])
        rows = ""
        for action in actions:
            rows += (f'<tr><td>{action.get("action", "-")}</td>'
                     f'<td>{action.get("impact", "-")}</td>'
                     f'<td>{action.get("status", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Actions to be documented</em></td></tr>'
        return (
            f'<div class="section">\n'
            f'  <h2>Key Actions</h2>\n'
            f'  <table><tr><th>Action</th><th>Impact</th><th>Status</th></tr>\n{rows}</table>\n'
            f'</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts} | '
            f'Race to Zero Campaign - UNFCCC High-Level Champions'
            f'</div>'
        )
