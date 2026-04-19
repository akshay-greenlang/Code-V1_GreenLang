# -*- coding: utf-8 -*-
"""
VarianceAnalysisReportTemplate - Emissions decomposition and variance attribution for PACK-022.

Renders a variance analysis report with Kaya-style decomposition (activity,
intensity, structural effects), driver attribution, waterfall data, cumulative
effects, rolling forecasts, and RAG alert status.

Sections:
    1. Period Summary
    2. Decomposition Results (activity/intensity/structural)
    3. Driver Attribution (top 10)
    4. Year-over-Year Waterfall Data
    5. Cumulative Effects Since Base Year
    6. Rolling Forecast (1-3yr)
    7. Alert Status (RAG with thresholds)
    8. Corrective Actions Required

Author: GreenLang Team
Version: 22.0.0
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

_MODULE_VERSION = "22.0.0"

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

class VarianceAnalysisReportTemplate:
    """
    Emissions decomposition and variance attribution report template.

    Decomposes emissions changes into activity, intensity, and structural
    effects. Attributes variances to top drivers, provides waterfall data,
    cumulative tracking, and rolling forecasts with RAG alerts.

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
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_period_summary(data),
            self._md_decomposition(data),
            self._md_driver_attribution(data),
            self._md_waterfall(data),
            self._md_cumulative_effects(data),
            self._md_rolling_forecast(data),
            self._md_alert_status(data),
            self._md_corrective_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_period_summary(data),
            self._html_decomposition(data),
            self._html_driver_attribution(data),
            self._html_waterfall(data),
            self._html_cumulative_effects(data),
            self._html_rolling_forecast(data),
            self._html_alert_status(data),
            self._html_corrective_actions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Variance Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        period = data.get("period", {})
        decomposition = data.get("decomposition", [])
        drivers = data.get("drivers", [])
        waterfall = data.get("waterfall", [])
        cumulative = data.get("cumulative_effects", [])
        forecast = data.get("rolling_forecast", [])
        alerts = data.get("alerts", [])
        actions = data.get("corrective_actions", [])

        result: Dict[str, Any] = {
            "template": "variance_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "period": period,
            "decomposition": decomposition,
            "drivers": drivers,
            "waterfall": waterfall,
            "cumulative_effects": cumulative,
            "rolling_forecast": forecast,
            "alerts": alerts,
            "corrective_actions": actions,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Variance Analysis Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_period_summary(self, data: Dict[str, Any]) -> str:
        period = data.get("period", {})
        prev = Decimal(str(period.get("previous_tco2e", 0)))
        curr = Decimal(str(period.get("current_tco2e", 0)))
        change = curr - prev
        pct = Decimal("0")
        if prev != 0:
            pct = (change / prev * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        target = Decimal(str(period.get("target_tco2e", 0)))
        gap = curr - target
        return (
            "## 1. Period Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Previous Period | {_dec_comma(prev)} tCO2e |\n"
            f"| Current Period | {_dec_comma(curr)} tCO2e |\n"
            f"| Absolute Change | {_dec_comma(change)} tCO2e |\n"
            f"| Relative Change | {_dec(pct)}% |\n"
            f"| Target | {_dec_comma(target)} tCO2e |\n"
            f"| Gap to Target | {_dec_comma(gap)} tCO2e |\n"
            f"| Period | {period.get('from', 'N/A')} to {period.get('to', 'N/A')} |"
        )

    def _md_decomposition(self, data: Dict[str, Any]) -> str:
        decomp = data.get("decomposition", [])
        lines = [
            "## 2. Decomposition Results\n",
            "Kaya-style decomposition of emissions change into activity, intensity, and structural effects.\n",
            "| Effect | Impact (tCO2e) | Share (%) | Direction | Description |",
            "|--------|---------------:|:---------:|:---------:|-------------|",
        ]
        total_impact = sum(abs(Decimal(str(d.get("impact_tco2e", 0)))) for d in decomp)
        for d in decomp:
            impact = Decimal(str(d.get("impact_tco2e", 0)))
            share = Decimal("0")
            if total_impact != 0:
                share = (abs(impact) / total_impact * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            direction = "Increase" if impact > 0 else "Decrease" if impact < 0 else "Neutral"
            lines.append(
                f"| {d.get('effect', '-')} "
                f"| {_dec_comma(impact)} "
                f"| {_dec(share)}% "
                f"| {direction} "
                f"| {d.get('description', '-')} |"
            )
        if not decomp:
            lines.append("| _No decomposition data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_driver_attribution(self, data: Dict[str, Any]) -> str:
        drivers = data.get("drivers", [])[:10]
        lines = [
            "## 3. Driver Attribution (Top 10)\n",
            "| Rank | Driver | Category | Impact (tCO2e) | Magnitude (%) | Controllable |",
            "|:----:|--------|----------|---------------:|:-------------:|:------------:|",
        ]
        sorted_drivers = sorted(drivers, key=lambda x: abs(x.get("impact_tco2e", 0)), reverse=True)
        for i, drv in enumerate(sorted_drivers[:10], 1):
            controllable = "Yes" if drv.get("controllable", False) else "No"
            lines.append(
                f"| {i} | {drv.get('name', '-')} "
                f"| {drv.get('category', '-')} "
                f"| {_dec_comma(drv.get('impact_tco2e', 0))} "
                f"| {_dec(drv.get('magnitude_pct', 0))}% "
                f"| {controllable} |"
            )
        if not drivers:
            lines.append("| - | _No driver data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_waterfall(self, data: Dict[str, Any]) -> str:
        waterfall = data.get("waterfall", [])
        lines = [
            "## 4. Year-over-Year Waterfall Data\n",
            "| Step | Label | Value (tCO2e) | Running Total (tCO2e) | Type |",
            "|:----:|-------|:-------------:|----------------------:|------|",
        ]
        for i, step in enumerate(waterfall, 1):
            lines.append(
                f"| {i} | {step.get('label', '-')} "
                f"| {_dec_comma(step.get('value_tco2e', 0))} "
                f"| {_dec_comma(step.get('running_total_tco2e', 0))} "
                f"| {step.get('type', '-')} |"
            )
        if not waterfall:
            lines.append("| - | _No waterfall data_ | - | - | - |")
        return "\n".join(lines)

    def _md_cumulative_effects(self, data: Dict[str, Any]) -> str:
        cumulative = data.get("cumulative_effects", [])
        lines = [
            "## 5. Cumulative Effects Since Base Year\n",
            "| Year | Activity Effect (tCO2e) | Intensity Effect (tCO2e) | Structural Effect (tCO2e) | Net Change (tCO2e) |",
            "|:----:|------------------------:|-------------------------:|--------------------------:|-------------------:|",
        ]
        for row in cumulative:
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec_comma(row.get('activity_effect_tco2e', 0))} "
                f"| {_dec_comma(row.get('intensity_effect_tco2e', 0))} "
                f"| {_dec_comma(row.get('structural_effect_tco2e', 0))} "
                f"| {_dec_comma(row.get('net_change_tco2e', 0))} |"
            )
        if not cumulative:
            lines.append("| - | _No cumulative data_ | - | - | - |")
        return "\n".join(lines)

    def _md_rolling_forecast(self, data: Dict[str, Any]) -> str:
        forecast = data.get("rolling_forecast", [])
        lines = [
            "## 6. Rolling Forecast (1-3yr)\n",
            "| Year | Forecast (tCO2e) | Target (tCO2e) | Gap (tCO2e) | Probability (%) | On Track |",
            "|:----:|:----------------:|--------------:|:------------:|:---------------:|:--------:|",
        ]
        for row in forecast:
            on_track = "YES" if row.get("on_track", False) else "NO"
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec_comma(row.get('forecast_tco2e', 0))} "
                f"| {_dec_comma(row.get('target_tco2e', 0))} "
                f"| {_dec_comma(row.get('gap_tco2e', 0))} "
                f"| {_dec(row.get('probability_pct', 0))}% "
                f"| {on_track} |"
            )
        if not forecast:
            lines.append("| - | _No forecast data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_alert_status(self, data: Dict[str, Any]) -> str:
        alerts = data.get("alerts", [])
        lines = [
            "## 7. Alert Status (RAG)\n",
            "| Alert | Metric | Threshold | Current | Status | Action Required |",
            "|-------|--------|-----------|---------|:------:|:---------------:|",
        ]
        for alert in alerts:
            status = alert.get("status", "AMBER").upper()
            lines.append(
                f"| {alert.get('name', '-')} "
                f"| {alert.get('metric', '-')} "
                f"| {alert.get('threshold', '-')} "
                f"| {alert.get('current', '-')} "
                f"| {status} "
                f"| {alert.get('action_required', 'No')} |"
            )
        if not alerts:
            lines.append("| _No alerts_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        lines = ["## 8. Corrective Actions Required\n"]
        if actions:
            lines.append("| # | Action | Owner | Deadline | Priority | Expected Impact (tCO2e) | Status |")
            lines.append("|---|--------|-------|----------|:--------:|------------------------:|:------:|")
            for i, act in enumerate(actions, 1):
                lines.append(
                    f"| {i} | {act.get('action', '-')} "
                    f"| {act.get('owner', '-')} "
                    f"| {act.get('deadline', '-')} "
                    f"| {act.get('priority', '-')} "
                    f"| {_dec_comma(act.get('expected_impact_tco2e', 0))} "
                    f"| {act.get('status', '-')} |"
                )
        else:
            lines.append("_No corrective actions required at this time._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Decomposition methodology: Logarithmic Mean Divisia Index (LMDI).*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".card-red{border-left-color:#c62828;background:linear-gradient(135deg,#ffebee,#ffcdd2);}"
            ".card-red .card-value{color:#c62828;}"
            ".card-amber{border-left-color:#e65100;background:linear-gradient(135deg,#fff3e0,#ffe0b2);}"
            ".card-amber .card-value{color:#e65100;}"
            ".rag-green{color:#1b5e20;font-weight:700;}"
            ".rag-amber{color:#e65100;font-weight:700;}"
            ".rag-red{color:#c62828;font-weight:700;}"
            ".increase{color:#c62828;}"
            ".decrease{color:#1b5e20;}"
            ".waterfall-pos{background:#ef5350;color:#fff;padding:4px 8px;border-radius:4px;}"
            ".waterfall-neg{background:#43a047;color:#fff;padding:4px 8px;border-radius:4px;}"
            ".waterfall-total{background:#1565c0;color:#fff;padding:4px 8px;border-radius:4px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Variance Analysis Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_period_summary(self, data: Dict[str, Any]) -> str:
        period = data.get("period", {})
        prev = Decimal(str(period.get("previous_tco2e", 0)))
        curr = Decimal(str(period.get("current_tco2e", 0)))
        change = curr - prev
        pct = Decimal("0")
        if prev != 0:
            pct = (change / prev * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        change_cls = "card-red" if change > 0 else ""
        pct_cls = "increase" if change > 0 else "decrease"
        return (
            f'<h2>1. Period Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Previous</div>'
            f'<div class="card-value">{_dec_comma(prev)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Current</div>'
            f'<div class="card-value">{_dec_comma(curr)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card {change_cls}"><div class="card-label">Change</div>'
            f'<div class="card-value">{_dec_comma(change)}</div>'
            f'<div class="card-unit">tCO2e (<span class="{pct_cls}">{_dec(pct)}%</span>)</div></div>\n'
            f'  <div class="card"><div class="card-label">Target</div>'
            f'<div class="card-value">{_dec_comma(period.get("target_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_decomposition(self, data: Dict[str, Any]) -> str:
        decomp = data.get("decomposition", [])
        total_impact = sum(abs(Decimal(str(d.get("impact_tco2e", 0)))) for d in decomp)
        rows = ""
        for d in decomp:
            impact = Decimal(str(d.get("impact_tco2e", 0)))
            share = Decimal("0")
            if total_impact != 0:
                share = (abs(impact) / total_impact * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            cls = "increase" if impact > 0 else "decrease"
            direction = "Increase" if impact > 0 else "Decrease" if impact < 0 else "Neutral"
            rows += (
                f'<tr><td>{d.get("effect", "-")}</td>'
                f'<td class="{cls}">{_dec_comma(impact)}</td>'
                f'<td>{_dec(share)}%</td>'
                f'<td>{direction}</td>'
                f'<td>{d.get("description", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Decomposition Results</h2>\n'
            f'<table>\n'
            f'<tr><th>Effect</th><th>Impact (tCO2e)</th><th>Share</th>'
            f'<th>Direction</th><th>Description</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_driver_attribution(self, data: Dict[str, Any]) -> str:
        drivers = data.get("drivers", [])
        sorted_drivers = sorted(drivers, key=lambda x: abs(x.get("impact_tco2e", 0)), reverse=True)[:10]
        rows = ""
        for i, drv in enumerate(sorted_drivers, 1):
            impact = Decimal(str(drv.get("impact_tco2e", 0)))
            cls = "increase" if impact > 0 else "decrease"
            controllable = "&#10004;" if drv.get("controllable", False) else "&#10008;"
            rows += (
                f'<tr><td>{i}</td><td><strong>{drv.get("name", "-")}</strong></td>'
                f'<td>{drv.get("category", "-")}</td>'
                f'<td class="{cls}">{_dec_comma(impact)}</td>'
                f'<td>{_dec(drv.get("magnitude_pct", 0))}%</td>'
                f'<td>{controllable}</td></tr>\n'
            )
        return (
            f'<h2>3. Driver Attribution (Top 10)</h2>\n'
            f'<table>\n'
            f'<tr><th>Rank</th><th>Driver</th><th>Category</th>'
            f'<th>Impact (tCO2e)</th><th>Magnitude</th><th>Controllable</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_waterfall(self, data: Dict[str, Any]) -> str:
        waterfall = data.get("waterfall", [])
        rows = ""
        for i, step in enumerate(waterfall, 1):
            val = float(Decimal(str(step.get("value_tco2e", 0))))
            step_type = step.get("type", "delta")
            if step_type == "total":
                val_cls = "waterfall-total"
            elif val > 0:
                val_cls = "waterfall-pos"
            else:
                val_cls = "waterfall-neg"
            rows += (
                f'<tr><td>{i}</td><td>{step.get("label", "-")}</td>'
                f'<td><span class="{val_cls}">{_dec_comma(step.get("value_tco2e", 0))}</span></td>'
                f'<td>{_dec_comma(step.get("running_total_tco2e", 0))}</td>'
                f'<td>{step_type}</td></tr>\n'
            )
        return (
            f'<h2>4. Year-over-Year Waterfall</h2>\n'
            f'<table>\n'
            f'<tr><th>Step</th><th>Label</th><th>Value (tCO2e)</th>'
            f'<th>Running Total</th><th>Type</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cumulative_effects(self, data: Dict[str, Any]) -> str:
        cumulative = data.get("cumulative_effects", [])
        rows = ""
        for row in cumulative:
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec_comma(row.get("activity_effect_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("intensity_effect_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("structural_effect_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("net_change_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>5. Cumulative Effects Since Base Year</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Activity Effect</th><th>Intensity Effect</th>'
            f'<th>Structural Effect</th><th>Net Change (tCO2e)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_rolling_forecast(self, data: Dict[str, Any]) -> str:
        forecast = data.get("rolling_forecast", [])
        rows = ""
        for row in forecast:
            on_track = row.get("on_track", False)
            cls = "rag-green" if on_track else "rag-red"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec_comma(row.get("forecast_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("target_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("gap_tco2e", 0))}</td>'
                f'<td>{_dec(row.get("probability_pct", 0))}%</td>'
                f'<td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>6. Rolling Forecast (1-3yr)</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Forecast (tCO2e)</th><th>Target (tCO2e)</th>'
            f'<th>Gap</th><th>Probability</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_alert_status(self, data: Dict[str, Any]) -> str:
        alerts = data.get("alerts", [])
        rows = ""
        for alert in alerts:
            status = alert.get("status", "AMBER").upper()
            cls = (
                "rag-green" if status == "GREEN"
                else "rag-red" if status == "RED"
                else "rag-amber"
            )
            rows += (
                f'<tr><td>{alert.get("name", "-")}</td>'
                f'<td>{alert.get("metric", "-")}</td>'
                f'<td>{alert.get("threshold", "-")}</td>'
                f'<td>{alert.get("current", "-")}</td>'
                f'<td class="{cls}">{status}</td>'
                f'<td>{alert.get("action_required", "No")}</td></tr>\n'
            )
        return (
            f'<h2>7. Alert Status (RAG)</h2>\n'
            f'<table>\n'
            f'<tr><th>Alert</th><th>Metric</th><th>Threshold</th>'
            f'<th>Current</th><th>Status</th><th>Action Required</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        rows = ""
        for i, act in enumerate(actions, 1):
            priority = act.get("priority", "MEDIUM")
            p_cls = (
                "rag-red" if priority.upper() == "HIGH"
                else "rag-green" if priority.upper() == "LOW"
                else "rag-amber"
            )
            rows += (
                f'<tr><td>{i}</td><td>{act.get("action", "-")}</td>'
                f'<td>{act.get("owner", "-")}</td>'
                f'<td>{act.get("deadline", "-")}</td>'
                f'<td class="{p_cls}">{priority}</td>'
                f'<td>{_dec_comma(act.get("expected_impact_tco2e", 0))}</td>'
                f'<td>{act.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>8. Corrective Actions Required</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Owner</th><th>Deadline</th>'
            f'<th>Priority</th><th>Expected Impact (tCO2e)</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Decomposition methodology: LMDI (Logarithmic Mean Divisia Index).</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
