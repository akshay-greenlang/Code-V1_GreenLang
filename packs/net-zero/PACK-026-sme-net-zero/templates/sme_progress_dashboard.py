# -*- coding: utf-8 -*-
"""
SMEProgressDashboardTemplate - Annual KPI tracking dashboard for PACK-026.

Renders an annual progress dashboard for SMEs with year-over-year
comparison, progress bars to 2030/2050 targets, scope trends,
quick wins implementation status, grant funding tracker, cost savings,
and next quarter actions.

Sections:
    1. KPI Summary Cards (actual vs target)
    2. Year-over-Year Comparison
    3. Progress Bars (% to 2030 + 2050 targets)
    4. Scope 1/2/3 Trends (last 3 years)
    5. Quick Wins Implementation Status
    6. Grant Funding Tracker
    7. Cost Savings (realized vs projected)
    8. Next Quarter Actions

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_progress_dashboard"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Status labels
# ---------------------------------------------------------------------------
_STATUS_MAP = {
    "completed": {"label": "Completed", "color": "#4caf50", "symbol": "[x]"},
    "in_progress": {"label": "In Progress", "color": "#ff9800", "symbol": "[~]"},
    "planned": {"label": "Planned", "color": "#9e9e9e", "symbol": "[ ]"},
    "delayed": {"label": "Delayed", "color": "#f44336", "symbol": "[!]"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default

def _status_info(status: str) -> Dict[str, str]:
    return _STATUS_MAP.get(status.lower(), _STATUS_MAP["planned"])

def _progress_bar_ascii(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + f"] {_dec(pct, 1)}%"

def _yoy_arrow(current: float, previous: float) -> str:
    if previous == 0:
        return "--"
    change = _safe_div(current - previous, abs(previous)) * 100
    if change < -1:
        return f"v {_pct(abs(change))} (good)"
    elif change > 1:
        return f"^ {_pct(change)} (increase)"
    return "-- flat"

# ===========================================================================
# Template Class
# ===========================================================================

class SMEProgressDashboardTemplate:
    """
    SME annual progress tracking dashboard template.

    Renders KPI cards, year-over-year comparison, target progress bars,
    scope trends, implementation status, grant tracker, cost savings,
    and next quarter actions across Markdown, HTML (interactive), and JSON.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the progress dashboard as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_summary(data),
            self._md_yoy_comparison(data),
            self._md_target_progress(data),
            self._md_scope_trends(data),
            self._md_implementation_status(data),
            self._md_grant_tracker(data),
            self._md_cost_savings(data),
            self._md_next_quarter(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the progress dashboard as interactive HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_summary(data),
            self._html_yoy_comparison(data),
            self._html_target_progress(data),
            self._html_scope_trends(data),
            self._html_implementation_status(data),
            self._html_grant_tracker(data),
            self._html_cost_savings(data),
            self._html_next_quarter(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Progress Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the progress dashboard as structured JSON."""
        self.generated_at = utcnow()
        current = float(data.get("current_tco2e", 0))
        baseline = float(data.get("baseline_tco2e", 0))
        target_2030 = float(data.get("target_2030_tco2e", 0))
        previous = float(data.get("previous_tco2e", 0))

        reduction_from_baseline = baseline - current
        pct_to_2030 = _safe_div(reduction_from_baseline,
                                baseline - target_2030) * 100 if (baseline - target_2030) > 0 else 0

        actions = data.get("actions", [])
        completed = sum(1 for a in actions if a.get("status", "").lower() == "completed")
        in_progress = sum(1 for a in actions if a.get("status", "").lower() == "in_progress")

        grants = data.get("grants", [])
        grant_received = sum(float(g.get("received", 0)) for g in grants)
        grant_applied = sum(float(g.get("applied", 0)) for g in grants)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {"name": data.get("org_name", "")},
            "reporting_year": data.get("reporting_year", ""),
            "kpis": {
                "current_emissions_tco2e": round(current, 2),
                "baseline_tco2e": round(baseline, 2),
                "previous_year_tco2e": round(previous, 2),
                "yoy_change_pct": round(_safe_div(current - previous, abs(previous)) * 100, 1) if previous else 0,
                "reduction_from_baseline_pct": round(_safe_div(reduction_from_baseline, baseline) * 100, 1),
                "progress_to_2030_pct": round(min(pct_to_2030, 100), 1),
                "target_2030_tco2e": round(target_2030, 2),
            },
            "scope_trends": data.get("scope_trends", []),
            "actions_status": {
                "total": len(actions),
                "completed": completed,
                "in_progress": in_progress,
                "planned": len(actions) - completed - in_progress,
                "completion_pct": round(_safe_div(completed, len(actions)) * 100, 1) if actions else 0,
                "actions": actions,
            },
            "grant_funding": {
                "total_applied": round(grant_applied, 2),
                "total_received": round(grant_received, 2),
                "grants": grants,
            },
            "cost_savings": {
                "projected": data.get("projected_savings", 0),
                "realized": data.get("realized_savings", 0),
                "variance_pct": round(
                    _safe_div(
                        float(data.get("realized_savings", 0)) - float(data.get("projected_savings", 0)),
                        float(data.get("projected_savings", 1))
                    ) * 100, 1
                ),
            },
            "next_quarter_actions": data.get("next_quarter_actions", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Progress Dashboard\n\n"
            f"**Organization:** {data.get('org_name', 'Your Company')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_kpi_summary(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        baseline = float(data.get("baseline_tco2e", 0))
        previous = float(data.get("previous_tco2e", 0))
        target_2030 = float(data.get("target_2030_tco2e", 0))
        reduction = baseline - current
        yoy = _yoy_arrow(current, previous)

        return (
            f"## Key Performance Indicators\n\n"
            f"| KPI | Value | vs. Previous Year |\n"
            f"|-----|------:|:-----------------:|\n"
            f"| **Current Emissions** | **{_dec_comma(current)} tCO2e** | {yoy} |\n"
            f"| Baseline | {_dec_comma(baseline)} tCO2e | - |\n"
            f"| Reduction from Baseline | {_dec_comma(reduction)} tCO2e "
            f"({_pct(_safe_div(reduction, baseline) * 100)}) | |\n"
            f"| 2030 Target | {_dec_comma(target_2030)} tCO2e | |"
        )

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        previous = float(data.get("previous_tco2e", 0))
        target = float(data.get("target_tco2e", 0))

        change = current - previous
        on_track = current <= target

        lines = [
            "## Year-over-Year Comparison\n",
            f"| Year | Actual | Target | On Track? |",
            f"|------|-------:|-------:|:---------:|",
            f"| Previous | {_dec_comma(previous)} tCO2e | | |",
            f"| Current | {_dec_comma(current)} tCO2e | {_dec_comma(target)} tCO2e "
            f"| {'YES' if on_track else 'NO'} |",
            f"| Change | {'+' if change > 0 else ''}{_dec_comma(change)} tCO2e | | |",
        ]
        return "\n".join(lines)

    def _md_target_progress(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        baseline = float(data.get("baseline_tco2e", 0))
        target_2030 = float(data.get("target_2030_tco2e", 0))

        if baseline > target_2030:
            needed = baseline - target_2030
            achieved = baseline - current
            pct_2030 = min(_safe_div(achieved, needed) * 100, 100)
        else:
            pct_2030 = 100

        pct_2050 = min(pct_2030 * 0.42, 100)  # rough proxy

        lines = [
            "## Progress to Targets\n",
            f"**2030 Target ({_pct(data.get('target_2030_pct', 42))} reduction):**",
            f"{_progress_bar_ascii(pct_2030)}",
            f"",
            f"**2050 Target (Net Zero):**",
            f"{_progress_bar_ascii(pct_2050)}",
        ]
        return "\n".join(lines)

    def _md_scope_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("scope_trends", [])
        if not trends:
            return ""

        lines = [
            "## Scope Trends (Last 3 Years)\n",
            "| Year | Scope 1 | Scope 2 | Scope 3 | Total |",
            "|------|--------:|--------:|--------:|------:|",
        ]
        for t in trends:
            s1 = float(t.get("scope1", 0))
            s2 = float(t.get("scope2", 0))
            s3 = float(t.get("scope3", 0))
            lines.append(
                f"| {t.get('year', '')} "
                f"| {_dec_comma(s1)} "
                f"| {_dec_comma(s2)} "
                f"| {_dec_comma(s3)} "
                f"| {_dec_comma(s1 + s2 + s3)} |"
            )
        return "\n".join(lines)

    def _md_implementation_status(self, data: Dict[str, Any]) -> str:
        actions = data.get("actions", [])
        if not actions:
            return ""

        completed = sum(1 for a in actions if a.get("status", "").lower() == "completed")
        in_progress = sum(1 for a in actions if a.get("status", "").lower() == "in_progress")
        planned = len(actions) - completed - in_progress
        pct = _safe_div(completed, len(actions)) * 100

        lines = [
            "## Quick Wins Implementation Status\n",
            f"**Completed:** {completed} | **In Progress:** {in_progress} | "
            f"**Planned:** {planned} | **Overall:** {_pct(pct)}\n",
            f"{_progress_bar_ascii(pct)}\n",
            "| Action | Status | Reduction | Notes |",
            "|--------|:------:|----------:|-------|",
        ]
        for a in actions:
            st = _status_info(a.get("status", "planned"))
            lines.append(
                f"| {a.get('name', '')} "
                f"| {st['symbol']} {st['label']} "
                f"| {_dec_comma(a.get('reduction_tco2e', 0))} tCO2e "
                f"| {a.get('notes', '')} |"
            )
        return "\n".join(lines)

    def _md_grant_tracker(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        if not grants:
            return ""

        currency = data.get("currency", "GBP")
        total_applied = sum(float(g.get("applied", 0)) for g in grants)
        total_received = sum(float(g.get("received", 0)) for g in grants)

        lines = [
            "## Grant Funding Tracker\n",
            f"**Applied:** {currency} {_dec_comma(total_applied)} | "
            f"**Received:** {currency} {_dec_comma(total_received)} | "
            f"**Success Rate:** {_pct(_safe_div(total_received, total_applied) * 100)}\n",
            "| Grant | Applied | Received | Status |",
            "|-------|--------:|---------:|:------:|",
        ]
        for g in grants:
            lines.append(
                f"| {g.get('name', '')} "
                f"| {currency} {_dec_comma(g.get('applied', 0))} "
                f"| {currency} {_dec_comma(g.get('received', 0))} "
                f"| {g.get('status', '')} |"
            )
        return "\n".join(lines)

    def _md_cost_savings(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        projected = float(data.get("projected_savings", 0))
        realized = float(data.get("realized_savings", 0))
        variance = realized - projected

        lines = [
            "## Cost Savings\n",
            f"| Metric | Amount |",
            f"|--------|-------:|",
            f"| Projected Savings | {currency} {_dec_comma(projected)} |",
            f"| Realized Savings | {currency} {_dec_comma(realized)} |",
            f"| Variance | {'+' if variance >= 0 else ''}{currency} {_dec_comma(variance)} "
            f"({_pct(_safe_div(variance, projected) * 100)}) |",
        ]
        return "\n".join(lines)

    def _md_next_quarter(self, data: Dict[str, Any]) -> str:
        actions = data.get("next_quarter_actions", [])
        if not actions:
            return ""

        lines = ["## Next Quarter Actions\n"]
        for idx, a in enumerate(actions, 1):
            lines.append(
                f"**{idx}. {a.get('action', '')}**  \n"
                f"   Owner: {a.get('owner', 'TBD')} | "
                f"Deadline: {a.get('deadline', 'TBD')} | "
                f"Expected Impact: {a.get('impact', '')}"
            )
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Annual progress tracking dashboard for SMEs.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:32px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".kpi-card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});"
            f"border-radius:10px;padding:14px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".kpi-label{{font-size:0.7em;color:#558b2f;text-transform:uppercase;}}"
            f".kpi-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".kpi-change{{font-size:0.75em;margin-top:2px;}}"
            f".kpi-good{{color:#4caf50;}}"
            f".kpi-bad{{color:#f44336;}}"
            f".kpi-neutral{{color:#9e9e9e;}}"
            f".progress-container{{margin:12px 0;}}"
            f".progress-label{{font-weight:600;margin-bottom:4px;}}"
            f".progress-track{{height:28px;background:#e0e0e0;border-radius:8px;overflow:hidden;"
            f"position:relative;}}"
            f".progress-fill{{height:100%;border-radius:8px;display:flex;align-items:center;"
            f"padding-left:10px;color:#fff;font-weight:600;font-size:0.85em;"
            f"transition:width 0.5s ease;}}"
            f".progress-2030{{background:linear-gradient(90deg,{_ACCENT},{_SECONDARY});}}"
            f".progress-2050{{background:linear-gradient(90deg,#66bb6a,#a5d6a7);}}"
            f".status-completed{{color:#4caf50;font-weight:600;}}"
            f".status-in_progress{{color:#ff9800;font-weight:600;}}"
            f".status-planned{{color:#9e9e9e;}}"
            f".status-delayed{{color:#f44336;font-weight:600;}}"
            f".action-card{{background:{_LIGHTER};border-left:4px solid {_ACCENT};"
            f"padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;}}"
            f".action-num{{display:inline-block;background:{_PRIMARY};color:#fff;width:24px;"
            f"height:24px;border-radius:50%;text-align:center;line-height:24px;"
            f"font-weight:700;font-size:0.8em;margin-right:6px;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.kpi-grid{{grid-template-columns:1fr 1fr;}}"
            f".report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Progress Dashboard</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'{data.get("reporting_year", "")} | Generated: {ts}</p>'
        )

    def _html_kpi_summary(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        baseline = float(data.get("baseline_tco2e", 0))
        previous = float(data.get("previous_tco2e", 0))
        reduction = baseline - current
        yoy_pct = _safe_div(current - previous, abs(previous)) * 100 if previous else 0
        red_pct = _safe_div(reduction, baseline) * 100

        yoy_cls = "kpi-good" if yoy_pct < 0 else ("kpi-bad" if yoy_pct > 0 else "kpi-neutral")
        yoy_label = f"{'v' if yoy_pct < 0 else '^'} {_pct(abs(yoy_pct))} YoY"

        return (
            f'<h2>Key Performance Indicators</h2>\n'
            f'<div class="kpi-grid">\n'
            f'  <div class="kpi-card"><div class="kpi-label">Current Emissions</div>'
            f'<div class="kpi-value">{_dec_comma(current)}</div>'
            f'<div class="kpi-change {yoy_cls}">{yoy_label}</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Baseline</div>'
            f'<div class="kpi-value">{_dec_comma(baseline)}</div>'
            f'<div class="kpi-change kpi-neutral">tCO2e</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Reduction</div>'
            f'<div class="kpi-value">{_dec_comma(reduction)}</div>'
            f'<div class="kpi-change kpi-good">{_pct(red_pct)} from baseline</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Previous Year</div>'
            f'<div class="kpi-value">{_dec_comma(previous)}</div>'
            f'<div class="kpi-change kpi-neutral">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        previous = float(data.get("previous_tco2e", 0))
        target = float(data.get("target_tco2e", 0))
        on_track = current <= target

        return (
            f'<h2>Year-over-Year Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>Period</th><th>Actual</th><th>Target</th><th>On Track?</th></tr>\n'
            f'<tr><td>Previous Year</td><td>{_dec_comma(previous)} tCO2e</td><td>-</td><td>-</td></tr>\n'
            f'<tr><td>Current Year</td><td>{_dec_comma(current)} tCO2e</td>'
            f'<td>{_dec_comma(target)} tCO2e</td>'
            f'<td class="{"status-completed" if on_track else "status-delayed"}">'
            f'{"YES" if on_track else "NO"}</td></tr>\n'
            f'<tr><td>Change</td><td>{_dec_comma(current - previous)} tCO2e</td>'
            f'<td>-</td><td>-</td></tr>\n'
            f'</table>'
        )

    def _html_target_progress(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_tco2e", 0))
        baseline = float(data.get("baseline_tco2e", 0))
        target_2030 = float(data.get("target_2030_tco2e", 0))

        if baseline > target_2030:
            needed = baseline - target_2030
            achieved = baseline - current
            pct_2030 = min(_safe_div(achieved, needed) * 100, 100)
        else:
            pct_2030 = 100

        pct_2050 = min(pct_2030 * 0.42, 100)

        return (
            f'<h2>Progress to Targets</h2>\n'
            f'<div class="progress-container">\n'
            f'  <div class="progress-label">2030 Target ({_pct(data.get("target_2030_pct", 42))} reduction)</div>\n'
            f'  <div class="progress-track">'
            f'<div class="progress-fill progress-2030" style="width:{max(pct_2030, 3):.1f}%">'
            f'{_pct(pct_2030)}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-container">\n'
            f'  <div class="progress-label">2050 Target (Net Zero)</div>\n'
            f'  <div class="progress-track">'
            f'<div class="progress-fill progress-2050" style="width:{max(pct_2050, 3):.1f}%">'
            f'{_pct(pct_2050)}</div></div>\n'
            f'</div>'
        )

    def _html_scope_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("scope_trends", [])
        if not trends:
            return ""

        rows = ""
        for t in trends:
            s1 = float(t.get("scope1", 0))
            s2 = float(t.get("scope2", 0))
            s3 = float(t.get("scope3", 0))
            rows += (
                f'<tr><td>{t.get("year", "")}</td>'
                f'<td>{_dec_comma(s1)}</td>'
                f'<td>{_dec_comma(s2)}</td>'
                f'<td>{_dec_comma(s3)}</td>'
                f'<td><strong>{_dec_comma(s1 + s2 + s3)}</strong></td></tr>\n'
            )

        return (
            f'<h2>Scope Trends (Last 3 Years)</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Scope 1</th><th>Scope 2</th>'
            f'<th>Scope 3</th><th>Total</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_implementation_status(self, data: Dict[str, Any]) -> str:
        actions = data.get("actions", [])
        if not actions:
            return ""

        completed = sum(1 for a in actions if a.get("status", "").lower() == "completed")
        pct = _safe_div(completed, len(actions)) * 100

        rows = ""
        for a in actions:
            st = _status_info(a.get("status", "planned"))
            rows += (
                f'<tr><td>{a.get("name", "")}</td>'
                f'<td class="status-{a.get("status", "planned").lower()}">{st["label"]}</td>'
                f'<td>{_dec_comma(a.get("reduction_tco2e", 0))} tCO2e</td>'
                f'<td>{a.get("notes", "")}</td></tr>\n'
            )

        return (
            f'<h2>Quick Wins Implementation Status</h2>\n'
            f'<div class="progress-container">\n'
            f'  <div class="progress-label">{completed}/{len(actions)} completed ({_pct(pct)})</div>\n'
            f'  <div class="progress-track">'
            f'<div class="progress-fill progress-2030" style="width:{max(pct, 3):.1f}%">'
            f'{_pct(pct)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Action</th><th>Status</th><th>Reduction</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_grant_tracker(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        if not grants:
            return ""

        currency = data.get("currency", "GBP")
        total_applied = sum(float(g.get("applied", 0)) for g in grants)
        total_received = sum(float(g.get("received", 0)) for g in grants)

        rows = ""
        for g in grants:
            rows += (
                f'<tr><td>{g.get("name", "")}</td>'
                f'<td>{currency} {_dec_comma(g.get("applied", 0))}</td>'
                f'<td>{currency} {_dec_comma(g.get("received", 0))}</td>'
                f'<td>{g.get("status", "")}</td></tr>\n'
            )

        return (
            f'<h2>Grant Funding Tracker</h2>\n'
            f'<div class="kpi-grid">\n'
            f'  <div class="kpi-card"><div class="kpi-label">Applied</div>'
            f'<div class="kpi-value">{currency} {_dec_comma(total_applied)}</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Received</div>'
            f'<div class="kpi-value">{currency} {_dec_comma(total_received)}</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Success Rate</div>'
            f'<div class="kpi-value">{_pct(_safe_div(total_received, total_applied) * 100)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Grant</th><th>Applied</th><th>Received</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cost_savings(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        projected = float(data.get("projected_savings", 0))
        realized = float(data.get("realized_savings", 0))
        variance = realized - projected

        return (
            f'<h2>Cost Savings</h2>\n'
            f'<div class="kpi-grid">\n'
            f'  <div class="kpi-card"><div class="kpi-label">Projected</div>'
            f'<div class="kpi-value">{currency} {_dec_comma(projected)}</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Realized</div>'
            f'<div class="kpi-value">{currency} {_dec_comma(realized)}</div></div>\n'
            f'  <div class="kpi-card"><div class="kpi-label">Variance</div>'
            f'<div class="kpi-value {"kpi-good" if variance >= 0 else "kpi-bad"}">'
            f'{"+" if variance >= 0 else ""}{currency} {_dec_comma(variance)}</div></div>\n'
            f'</div>'
        )

    def _html_next_quarter(self, data: Dict[str, Any]) -> str:
        actions = data.get("next_quarter_actions", [])
        if not actions:
            return ""

        cards = ""
        for idx, a in enumerate(actions, 1):
            cards += (
                f'<div class="action-card">'
                f'<span class="action-num">{idx}</span>'
                f'<strong>{a.get("action", "")}</strong><br>'
                f'<span style="font-size:0.8em;color:#689f38;">'
                f'Owner: {a.get("owner", "TBD")} | '
                f'Deadline: {a.get("deadline", "TBD")} | '
                f'Impact: {a.get("impact", "")}</span>'
                f'</div>\n'
            )

        return f'<h2>Next Quarter Actions</h2>\n{cards}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Annual progress tracking dashboard for SMEs'
            f'</div>'
        )
