# -*- coding: utf-8 -*-
"""
ProgressDashboardReportTemplate - Annual progress tracking for PACK-023.

Renders a comprehensive progress dashboard covering annual progress against
SBTi targets with trajectory analysis, RAG status for each target, actual vs
pathway comparison, annualized reduction rate vs required rate, emissions
budget remaining, year-over-year trend analysis, and corrective action
triggers with recommendations.

Sections:
    1. Progress Overview (headline metrics, RAG status)
    2. Target Progress Detail (per-target actual vs pathway)
    3. Annualized Reduction Rate Analysis
    4. Emissions Budget Tracker
    5. Year-over-Year Trend
    6. Corrective Action Triggers
    7. Recommendations & Next Steps

Author: GreenLang Team
Version: 23.0.0
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

_MODULE_VERSION = "23.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
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

def _pct(val: Any) -> str:
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _rag_status(status: str) -> str:
    """Normalize RAG status for display."""
    s = str(status).upper()
    if s in ("GREEN", "G", "ON_TRACK"):
        return "GREEN"
    elif s in ("AMBER", "A", "AT_RISK"):
        return "AMBER"
    elif s in ("RED", "R", "OFF_TRACK"):
        return "RED"
    return status

def _rag_description(status: str) -> str:
    """Return description for RAG status."""
    s = _rag_status(status)
    mapping = {
        "GREEN": "On track - actual reduction meets or exceeds pathway",
        "AMBER": "At risk - actual reduction within 10% of pathway",
        "RED": "Off track - actual reduction more than 10% behind pathway",
    }
    return mapping.get(s, "Status unknown")

class ProgressDashboardReportTemplate:
    """
    Annual progress dashboard report template for SBTi alignment.

    Renders a comprehensive progress tracking dashboard covering actual vs
    target pathway comparison, RAG status per target, annualized reduction
    rate analysis, emissions budget tracking, year-over-year trends, and
    corrective action triggers with recommendations.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgressDashboardReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render progress dashboard report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_progress_overview(data),
            self._md_target_progress_detail(data),
            self._md_reduction_rate_analysis(data),
            self._md_budget_tracker(data),
            self._md_yoy_trend(data),
            self._md_corrective_actions(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render progress dashboard report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_progress_overview(data),
            self._html_target_progress_detail(data),
            self._html_reduction_rate_analysis(data),
            self._html_budget_tracker(data),
            self._html_yoy_trend(data),
            self._html_corrective_actions(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Progress Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render progress dashboard report as structured JSON."""
        self.generated_at = utcnow()
        targets = data.get("targets", [])
        budget = data.get("budget", {})
        yoy = data.get("yoy_trend", [])
        triggers = data.get("corrective_triggers", [])

        green = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "GREEN"])
        amber = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "AMBER"])
        red = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "RED"])

        result: Dict[str, Any] = {
            "template": "progress_dashboard_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "progress_overview": {
                "total_targets": len(targets),
                "green": green,
                "amber": amber,
                "red": red,
                "overall_status": data.get("overall_rag", "N/A"),
            },
            "targets": targets,
            "reduction_rates": data.get("reduction_rates", []),
            "budget": budget,
            "yoy_trend": yoy,
            "corrective_triggers": triggers,
            "recommendations": data.get("recommendations", []),
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
            f"# SBTi Progress Dashboard\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_progress_overview(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        green = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "GREEN"])
        amber = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "AMBER"])
        red = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "RED"])
        overall = data.get("overall_rag", "N/A")
        total_actual = float(data.get("total_actual_tco2e", 0))
        total_pathway = float(data.get("total_pathway_tco2e", 0))
        variance = total_actual - total_pathway if total_pathway > 0 else 0
        variance_pct = (variance / total_pathway * 100) if total_pathway > 0 else 0

        return (
            f"## 1. Progress Overview\n\n"
            f"**Overall RAG Status:** {_rag_status(overall)}\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Targets Tracked | {len(targets)} |\n"
            f"| GREEN (On Track) | {green} |\n"
            f"| AMBER (At Risk) | {amber} |\n"
            f"| RED (Off Track) | {red} |\n"
            f"| Actual Emissions (tCO2e) | {_dec_comma(total_actual, 0)} |\n"
            f"| Pathway Target (tCO2e) | {_dec_comma(total_pathway, 0)} |\n"
            f"| Variance (tCO2e) | {'+' if variance > 0 else ''}{_dec_comma(variance, 0)} |\n"
            f"| Variance (%) | {'+' if variance_pct > 0 else ''}{_pct(variance_pct)} |\n\n"
            f"**RAG Legend:**\n"
            f"- **GREEN:** {_rag_description('GREEN')}\n"
            f"- **AMBER:** {_rag_description('AMBER')}\n"
            f"- **RED:** {_rag_description('RED')}"
        )

    def _md_target_progress_detail(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 2. Target Progress Detail\n",
            "Per-target actual vs pathway comparison with RAG status.\n",
            "| # | Target | Scope | Base Year (tCO2e) | Actual (tCO2e) "
            "| Pathway (tCO2e) | Variance (%) | Cumul. Reduction (%) "
            "| Required (%) | RAG |",
            "|---|--------|-------|------------------:|:--------------:"
            "|:---------------:|:------------:|:--------------------:"
            "|:------------:|:---:|",
        ]
        for i, t in enumerate(targets, 1):
            actual = float(t.get("actual_tco2e", 0))
            pathway = float(t.get("pathway_tco2e", 0))
            variance_pct = (
                (actual - pathway) / pathway * 100
                if pathway > 0 else 0
            )
            lines.append(
                f"| {i} | {t.get('target_name', '-')} "
                f"| {t.get('scope', '-')} "
                f"| {_dec_comma(t.get('base_year_tco2e', 0), 0)} "
                f"| {_dec_comma(actual, 0)} "
                f"| {_dec_comma(pathway, 0)} "
                f"| {'+' if variance_pct > 0 else ''}{_pct(variance_pct)} "
                f"| {_pct(t.get('cumulative_reduction_pct', 0))} "
                f"| {_pct(t.get('required_reduction_pct', 0))} "
                f"| {_rag_status(t.get('rag_status', ''))} |"
            )
        if not targets:
            lines.append(
                "| - | _No targets tracked_ | - | - | - | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_reduction_rate_analysis(self, data: Dict[str, Any]) -> str:
        rates = data.get("reduction_rates", [])
        lines = [
            "## 3. Annualized Reduction Rate Analysis\n",
            "Comparison of actual annualized reduction rate (ARR) against "
            "the required rate for each target.\n",
            "| Target | Scope | Required ARR (%/yr) | Actual ARR (%/yr) "
            "| Gap (%/yr) | On Pace | Years to Target |",
            "|--------|-------|:-------------------:|:-----------------:"
            "|:----------:|:-------:|:---------------:|",
        ]
        for r in rates:
            required = float(r.get("required_arr", 0))
            actual = float(r.get("actual_arr", 0))
            gap = actual - required
            on_pace = "Yes" if gap >= 0 else "No"
            lines.append(
                f"| {r.get('target_name', '-')} "
                f"| {r.get('scope', '-')} "
                f"| {_pct(required)} "
                f"| {_pct(actual)} "
                f"| {'+' if gap > 0 else ''}{_pct(gap)} "
                f"| {on_pace} "
                f"| {r.get('years_to_target', '-')} |"
            )
        if not rates:
            lines.append(
                "| - | _No rates computed_ | - | - | - | - | - |"
            )

        # Summary
        if rates:
            avg_required = sum(float(r.get("required_arr", 0)) for r in rates) / len(rates)
            avg_actual = sum(float(r.get("actual_arr", 0)) for r in rates) / len(rates)
            lines.append("")
            lines.append(
                f"**Average Required ARR:** {_pct(avg_required)}  \n"
                f"**Average Actual ARR:** {_pct(avg_actual)}  \n"
                f"**Overall Pace:** "
                f"{'On Track' if avg_actual >= avg_required else 'Behind Schedule'}"
            )
        return "\n".join(lines)

    def _md_budget_tracker(self, data: Dict[str, Any]) -> str:
        budget = data.get("budget", {})
        targets_budget = budget.get("targets", [])
        lines = [
            "## 4. Emissions Budget Tracker\n",
            "Remaining emissions budget from base year to target year.\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Budget (base to target) | {_dec_comma(budget.get('total_budget_tco2e', 0), 0)} tCO2e |\n"
            f"| Budget Used | {_dec_comma(budget.get('used_tco2e', 0), 0)} tCO2e |\n"
            f"| Budget Remaining | {_dec_comma(budget.get('remaining_tco2e', 0), 0)} tCO2e |\n"
            f"| Budget Used (%) | {_pct(budget.get('used_pct', 0))} |\n"
            f"| Time Elapsed (%) | {_pct(budget.get('time_elapsed_pct', 0))} |\n"
            f"| Budget vs Time | {budget.get('budget_vs_time', 'N/A')} |",
        ]

        if targets_budget:
            lines.append("")
            lines.append("### Per-Target Budget\n")
            lines.append(
                "| Target | Scope | Total Budget | Used | Remaining "
                "| Used (%) | Status |"
            )
            lines.append(
                "|--------|-------|:------------:|:----:|:---------:"
                "|:--------:|--------|"
            )
            for tb in targets_budget:
                lines.append(
                    f"| {tb.get('target_name', '-')} "
                    f"| {tb.get('scope', '-')} "
                    f"| {_dec_comma(tb.get('total_budget_tco2e', 0), 0)} "
                    f"| {_dec_comma(tb.get('used_tco2e', 0), 0)} "
                    f"| {_dec_comma(tb.get('remaining_tco2e', 0), 0)} "
                    f"| {_pct(tb.get('used_pct', 0))} "
                    f"| {tb.get('status', '-')} |"
                )

        return "\n".join(lines)

    def _md_yoy_trend(self, data: Dict[str, Any]) -> str:
        yoy = data.get("yoy_trend", [])
        lines = [
            "## 5. Year-over-Year Trend\n",
            "Historical emissions trend with year-over-year changes.\n",
            "| Year | Total Emissions (tCO2e) | S1+S2 (tCO2e) | S3 (tCO2e) "
            "| YoY Change (%) | Cumul. Change (%) | Pathway (tCO2e) "
            "| Status |",
            "|:----:|:----------------------:|:-------------:|:----------:"
            "|:--------------:|:-----------------:|:---------------:"
            "|--------|",
        ]
        for y in yoy:
            lines.append(
                f"| {y.get('year', '-')} "
                f"| {_dec_comma(y.get('total_tco2e', 0), 0)} "
                f"| {_dec_comma(y.get('s1s2_tco2e', 0), 0)} "
                f"| {_dec_comma(y.get('s3_tco2e', 0), 0)} "
                f"| {'+' if float(y.get('yoy_change_pct', 0)) > 0 else ''}"
                f"{_pct(y.get('yoy_change_pct', 0))} "
                f"| {'+' if float(y.get('cumul_change_pct', 0)) > 0 else ''}"
                f"{_pct(y.get('cumul_change_pct', 0))} "
                f"| {_dec_comma(y.get('pathway_tco2e', 0), 0)} "
                f"| {y.get('status', '-')} |"
            )
        if not yoy:
            lines.append(
                "| - | _No trend data_ | - | - | - | - | - | - |"
            )

        # Trend summary
        trend_summary = data.get("trend_summary", {})
        if trend_summary:
            lines.append("")
            lines.append(
                f"**Base Year:** {trend_summary.get('base_year', 'N/A')} "
                f"({_dec_comma(trend_summary.get('base_year_tco2e', 0), 0)} tCO2e)  \n"
                f"**Latest Year:** {trend_summary.get('latest_year', 'N/A')} "
                f"({_dec_comma(trend_summary.get('latest_year_tco2e', 0), 0)} tCO2e)  \n"
                f"**Total Reduction:** {_pct(trend_summary.get('total_reduction_pct', 0))}  \n"
                f"**Average ARR:** {_pct(trend_summary.get('avg_arr', 0))} per year  \n"
                f"**Trend Direction:** {trend_summary.get('direction', 'N/A')}"
            )
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        triggers = data.get("corrective_triggers", [])
        lines = [
            "## 6. Corrective Action Triggers\n",
            "Automated triggers identifying when corrective action is required.\n",
            "| # | Trigger | Condition | Current Value "
            "| Threshold | Triggered | Severity |",
            "|---|---------|-----------|:-------------:"
            "|:---------:|:---------:|:--------:|",
        ]
        for i, t in enumerate(triggers, 1):
            triggered = "YES" if t.get("triggered", False) else "NO"
            lines.append(
                f"| {i} | {t.get('trigger', '-')} "
                f"| {t.get('condition', '-')} "
                f"| {t.get('current_value', '-')} "
                f"| {t.get('threshold', '-')} "
                f"| {triggered} "
                f"| {t.get('severity', '-')} |"
            )
        if not triggers:
            lines.append(
                "| - | _No triggers defined_ | - | - | - | - | - |"
            )

        # Default triggers if none provided
        default_triggers = data.get("default_trigger_rules", [])
        if default_triggers:
            lines.append("")
            lines.append("### Trigger Rules\n")
            lines.append("| Rule | Description | Auto-Action |")
            lines.append("|------|-------------|-------------|")
            for r in default_triggers:
                lines.append(
                    f"| {r.get('rule', '-')} "
                    f"| {r.get('description', '-')} "
                    f"| {r.get('auto_action', '-')} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = [
            "## 7. Recommendations & Next Steps\n",
            "Prioritized actions to improve target alignment.\n",
            "| # | Recommendation | Target | Impact "
            "| Effort | Priority | Timeline | Owner |",
            "|---|----------------|--------|:------:"
            "|:------:|:--------:|:--------:|-------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('recommendation', '-')} "
                f"| {r.get('target', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('effort', '-')} "
                f"| {r.get('priority', '-')} "
                f"| {r.get('timeline', '-')} "
                f"| {r.get('owner', '-')} |"
            )
        if not recs:
            lines.append(
                "| - | _No recommendations_ | - | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Progress tracking per SBTi Monitoring, Reporting and Verification (MRV) "
            f"Guidance and Corporate Manual V5.3.*"
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
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".rag-green{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 10px;font-size:0.85em;font-weight:600;}"
            ".rag-amber{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 10px;font-size:0.85em;font-weight:600;}"
            ".rag-red{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 10px;font-size:0.85em;font-weight:600;}"
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".variance-positive{color:#c62828;font-weight:700;}"
            ".variance-negative{color:#2e7d32;font-weight:700;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_rag_badge(self, status: str) -> str:
        """Return an HTML badge for RAG status."""
        s = _rag_status(status)
        if s == "GREEN":
            return '<span class="rag-green">GREEN</span>'
        elif s == "AMBER":
            return '<span class="rag-amber">AMBER</span>'
        elif s == "RED":
            return '<span class="rag-red">RED</span>'
        return status

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Progress Dashboard</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_progress_overview(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        green = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "GREEN"])
        amber = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "AMBER"])
        red = len([t for t in targets if _rag_status(t.get("rag_status", "")) == "RED"])
        overall = data.get("overall_rag", "N/A")
        total_actual = float(data.get("total_actual_tco2e", 0))
        total_pathway = float(data.get("total_pathway_tco2e", 0))
        variance = total_actual - total_pathway if total_pathway > 0 else 0
        var_cls = "variance-positive" if variance > 0 else "variance-negative"

        return (
            f'<h2>1. Progress Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall Status</div>'
            f'<div class="card-value">{self._html_rag_badge(overall)}</div></div>\n'
            f'  <div class="card"><div class="card-label">GREEN</div>'
            f'<div class="card-value">{green}</div></div>\n'
            f'  <div class="card"><div class="card-label">AMBER</div>'
            f'<div class="card-value">{amber}</div></div>\n'
            f'  <div class="card"><div class="card-label">RED</div>'
            f'<div class="card-value">{red}</div></div>\n'
            f'  <div class="card"><div class="card-label">Actual</div>'
            f'<div class="card-value">{_dec_comma(total_actual, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Variance</div>'
            f'<div class="card-value {var_cls}">'
            f'{"+" if variance > 0 else ""}{_dec_comma(variance, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_target_progress_detail(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for i, t in enumerate(targets, 1):
            actual = float(t.get("actual_tco2e", 0))
            pathway = float(t.get("pathway_tco2e", 0))
            variance_pct = (actual - pathway) / pathway * 100 if pathway > 0 else 0
            var_cls = "variance-positive" if variance_pct > 0 else "variance-negative"
            rows += (
                f'<tr><td>{i}</td>'
                f'<td><strong>{t.get("target_name", "-")}</strong></td>'
                f'<td>{t.get("scope", "-")}</td>'
                f'<td>{_dec_comma(t.get("base_year_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(actual, 0)}</td>'
                f'<td>{_dec_comma(pathway, 0)}</td>'
                f'<td class="{var_cls}">{"+" if variance_pct > 0 else ""}'
                f'{_pct(variance_pct)}</td>'
                f'<td>{_pct(t.get("cumulative_reduction_pct", 0))}</td>'
                f'<td>{_pct(t.get("required_reduction_pct", 0))}</td>'
                f'<td>{self._html_rag_badge(t.get("rag_status", ""))}</td></tr>\n'
            )
        return (
            f'<h2>2. Target Progress Detail</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Target</th><th>Scope</th><th>Base Year</th>'
            f'<th>Actual</th><th>Pathway</th><th>Variance</th>'
            f'<th>Cumul. Red.</th><th>Required</th><th>RAG</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_reduction_rate_analysis(self, data: Dict[str, Any]) -> str:
        rates = data.get("reduction_rates", [])
        rows = ""
        for r in rates:
            required = float(r.get("required_arr", 0))
            actual = float(r.get("actual_arr", 0))
            gap = actual - required
            on_pace = actual >= required
            gap_cls = "variance-negative" if gap >= 0 else "variance-positive"
            gap_sign = "+" if gap > 0 else ""
            pace_badge = (
                '<span class="badge-pass">Yes</span>'
                if on_pace
                else '<span class="badge-fail">No</span>'
            )
            rows += (
                f'<tr><td>{r.get("target_name", "-")}</td>'
                f'<td>{r.get("scope", "-")}</td>'
                f'<td>{_pct(required)}</td>'
                f'<td>{_pct(actual)}</td>'
                f'<td class="{gap_cls}">{gap_sign}{_pct(gap)}</td>'
                f'<td>{pace_badge}</td>'
                f'<td>{r.get("years_to_target", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Annualized Reduction Rate Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>Target</th><th>Scope</th><th>Required ARR</th>'
            f'<th>Actual ARR</th><th>Gap</th><th>On Pace</th>'
            f'<th>Years Left</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_budget_tracker(self, data: Dict[str, Any]) -> str:
        budget = data.get("budget", {})
        used_pct = float(budget.get("used_pct", 0))
        time_pct = float(budget.get("time_elapsed_pct", 0))
        budget_color = "#43a047" if used_pct <= time_pct else "#ff9800" if used_pct <= time_pct * 1.1 else "#ef5350"

        targets_budget = budget.get("targets", [])
        target_rows = ""
        for tb in targets_budget:
            target_rows += (
                f'<tr><td>{tb.get("target_name", "-")}</td>'
                f'<td>{tb.get("scope", "-")}</td>'
                f'<td>{_dec_comma(tb.get("total_budget_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(tb.get("used_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(tb.get("remaining_tco2e", 0), 0)}</td>'
                f'<td>{_pct(tb.get("used_pct", 0))}</td>'
                f'<td>{tb.get("status", "-")}</td></tr>\n'
            )
        target_html = ""
        if targets_budget:
            target_html = (
                f'<h3>Per-Target Budget</h3>\n'
                f'<table><tr><th>Target</th><th>Scope</th><th>Total Budget</th>'
                f'<th>Used</th><th>Remaining</th><th>Used %</th><th>Status</th></tr>\n'
                f'{target_rows}</table>\n'
            )

        return (
            f'<h2>4. Emissions Budget Tracker</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Budget</div>'
            f'<div class="card-value">{_dec_comma(budget.get("total_budget_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Used</div>'
            f'<div class="card-value">{_pct(used_pct)}</div>'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(used_pct, 100)}%;background:{budget_color};"></div></div></div>\n'
            f'  <div class="card"><div class="card-label">Remaining</div>'
            f'<div class="card-value">{_dec_comma(budget.get("remaining_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Time Elapsed</div>'
            f'<div class="card-value">{_pct(time_pct)}</div></div>\n'
            f'</div>\n'
            f'{target_html}'
        )

    def _html_yoy_trend(self, data: Dict[str, Any]) -> str:
        yoy = data.get("yoy_trend", [])
        rows = ""
        for y in yoy:
            yoy_change = float(y.get("yoy_change_pct", 0))
            cls = "variance-negative" if yoy_change < 0 else "variance-positive" if yoy_change > 0 else ""
            rows += (
                f'<tr><td>{y.get("year", "-")}</td>'
                f'<td>{_dec_comma(y.get("total_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(y.get("s1s2_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(y.get("s3_tco2e", 0), 0)}</td>'
                f'<td class="{cls}">{"+" if yoy_change > 0 else ""}'
                f'{_pct(yoy_change)}</td>'
                f'<td>{_pct(y.get("cumul_change_pct", 0))}</td>'
                f'<td>{_dec_comma(y.get("pathway_tco2e", 0), 0)}</td>'
                f'<td>{y.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Year-over-Year Trend</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Total (tCO2e)</th><th>S1+S2</th>'
            f'<th>S3</th><th>YoY Change</th><th>Cumul.</th>'
            f'<th>Pathway</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        triggers = data.get("corrective_triggers", [])
        rows = ""
        for i, t in enumerate(triggers, 1):
            triggered = t.get("triggered", False)
            badge = (
                '<span class="badge-fail">YES</span>'
                if triggered
                else '<span class="badge-pass">NO</span>'
            )
            rows += (
                f'<tr><td>{i}</td><td>{t.get("trigger", "-")}</td>'
                f'<td>{t.get("condition", "-")}</td>'
                f'<td>{t.get("current_value", "-")}</td>'
                f'<td>{t.get("threshold", "-")}</td>'
                f'<td>{badge}</td>'
                f'<td>{t.get("severity", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Corrective Action Triggers</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Trigger</th><th>Condition</th>'
            f'<th>Current</th><th>Threshold</th><th>Triggered</th>'
            f'<th>Severity</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("recommendation", "-")}</td>'
                f'<td>{r.get("target", "-")}</td>'
                f'<td>{r.get("impact", "-")}</td>'
                f'<td>{r.get("effort", "-")}</td>'
                f'<td>{r.get("priority", "-")}</td>'
                f'<td>{r.get("timeline", "-")}</td>'
                f'<td>{r.get("owner", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Recommendations & Next Steps</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Recommendation</th><th>Target</th>'
            f'<th>Impact</th><th>Effort</th><th>Priority</th>'
            f'<th>Timeline</th><th>Owner</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Progress tracking per SBTi MRV Guidance and Corporate Manual V5.3.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
