# -*- coding: utf-8 -*-
"""
SMEBaselineReportTemplate - 1-2 page visual emissions baseline for PACK-026.

Renders a simplified GHG baseline dashboard optimized for SME audiences
with visual bar charts, executive summary, scope breakdown, industry
peer comparison, data quality badge, top emission sources, and next steps.

Sections:
    1. Executive Summary (total emissions, intensity per employee)
    2. Scope 1/2/3 Breakdown with simple bar charts
    3. Industry Peer Comparison (percentile ranking)
    4. Data Quality Score (Bronze/Silver/Gold)
    5. Top 3 Emission Sources
    6. Simple Next Steps

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
from typing import Any, Dict, List, Optional, Union

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_baseline_report"

# ---------------------------------------------------------------------------
# Green colour scheme
# ---------------------------------------------------------------------------
_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Data quality tiers
# ---------------------------------------------------------------------------
_DQ_TIERS = {
    "gold": {"label": "Gold", "min_score": 80, "color": "#ffd700", "emoji": "***"},
    "silver": {"label": "Silver", "min_score": 50, "color": "#c0c0c0", "emoji": "**"},
    "bronze": {"label": "Bronze", "min_score": 0, "color": "#cd7f32", "emoji": "*"},
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

def _dq_tier(score: float) -> Dict[str, Any]:
    """Return data quality tier for a numeric score 0-100."""
    if score >= 80:
        return _DQ_TIERS["gold"]
    elif score >= 50:
        return _DQ_TIERS["silver"]
    return _DQ_TIERS["bronze"]

def _ascii_bar(value: float, max_value: float, width: int = 30, char: str = "#") -> str:
    """Render a simple ASCII bar for markdown output."""
    if max_value <= 0:
        return ""
    filled = int(round(value / max_value * width))
    filled = max(0, min(width, filled))
    return char * filled + "." * (width - filled)

def _scope_color(scope: str) -> str:
    """Return CSS colour for scope."""
    mapping = {"scope1": _ACCENT, "scope2": "#66bb6a", "scope3": "#a5d6a7"}
    return mapping.get(scope.lower().replace(" ", ""), _ACCENT)

# ---------------------------------------------------------------------------
# Scope 3 simplified categories (SME-relevant)
# ---------------------------------------------------------------------------
_SME_SCOPE3 = [
    {"num": "1", "name": "Purchased Goods & Services"},
    {"num": "4", "name": "Upstream Transportation"},
    {"num": "5", "name": "Waste"},
    {"num": "6", "name": "Business Travel"},
    {"num": "7", "name": "Employee Commuting"},
]

# ===========================================================================
# Template Class
# ===========================================================================

class SMEBaselineReportTemplate:
    """
    SME-optimised 1-2 page GHG baseline dashboard template.

    Renders a visual emissions baseline report designed for small and
    medium enterprises, with simplified scope breakdown, peer comparison,
    data quality badges, and actionable next steps across four output
    formats: Markdown, HTML, JSON, and Excel-ready dict.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "excel"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the SME baseline report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_scope_breakdown(data),
            self._md_scope_bars(data),
            self._md_peer_comparison(data),
            self._md_data_quality(data),
            self._md_top_sources(data),
            self._md_next_steps(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the SME baseline report as HTML with inline CSS."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_scope_breakdown(data),
            self._html_peer_comparison(data),
            self._html_data_quality(data),
            self._html_top_sources(data),
            self._html_next_steps(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Emissions Baseline</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the SME baseline report as structured JSON."""
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)
        dq_score = float(data.get("data_quality_score", 0))
        tier = _dq_tier(dq_score)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "employees": employees,
                "revenue": data.get("revenue", 0),
                "currency": data.get("currency", "GBP"),
            },
            "reporting_year": data.get("reporting_year", ""),
            "emissions": {
                "total_tco2e": round(total, 2),
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "scope1_pct": round(_safe_div(s1, total) * 100, 1),
                "scope2_pct": round(_safe_div(s2, total) * 100, 1),
                "scope3_pct": round(_safe_div(s3, total) * 100, 1),
                "intensity_per_employee": round(intensity, 2),
                "intensity_per_revenue": round(
                    _safe_div(total, float(data.get("revenue", 1))) * 1_000_000, 2
                ),
            },
            "data_quality": {
                "score": dq_score,
                "tier": tier["label"],
                "details": data.get("data_quality_details", []),
            },
            "peer_comparison": {
                "sector": data.get("sector", ""),
                "sector_average_tco2e": data.get("sector_avg_tco2e", 0),
                "sector_median_tco2e": data.get("sector_median_tco2e", 0),
                "percentile": data.get("percentile_rank", 50),
                "best_in_class_tco2e": data.get("best_in_class_tco2e", 0),
            },
            "top_sources": data.get("top_sources", [])[:3],
            "next_steps": data.get("next_steps", []),
            "scope3_categories": data.get("scope3_categories", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure with worksheet definitions."""
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)
        dq_score = float(data.get("data_quality_score", 0))
        tier = _dq_tier(dq_score)

        summary_sheet = {
            "name": "Baseline Summary",
            "headers": ["Metric", "Value", "Unit"],
            "rows": [
                ["Organization", data.get("org_name", ""), ""],
                ["Reporting Year", data.get("reporting_year", ""), ""],
                ["Sector", data.get("sector", ""), ""],
                ["Employees", employees, "FTE"],
                ["Total Emissions", round(total, 2), "tCO2e"],
                ["Scope 1", round(s1, 2), "tCO2e"],
                ["Scope 2", round(s2, 2), "tCO2e"],
                ["Scope 3", round(s3, 2), "tCO2e"],
                ["Intensity (per employee)", round(intensity, 2), "tCO2e/FTE"],
                ["Data Quality Score", dq_score, f"/ 100 ({tier['label']})"],
                ["Sector Average", data.get("sector_avg_tco2e", 0), "tCO2e"],
                ["Percentile Rank", data.get("percentile_rank", 50), "%ile"],
            ],
        }

        scope_sheet = {
            "name": "Scope Breakdown",
            "headers": ["Scope", "Emissions (tCO2e)", "% of Total", "Key Sources"],
            "rows": [
                ["Scope 1 - Direct", round(s1, 2),
                 round(_safe_div(s1, total) * 100, 1),
                 data.get("scope1_sources_summary", "Gas, fleet, refrigerants")],
                ["Scope 2 - Electricity", round(s2, 2),
                 round(_safe_div(s2, total) * 100, 1),
                 data.get("scope2_sources_summary", "Purchased electricity, heat")],
                ["Scope 3 - Value Chain", round(s3, 2),
                 round(_safe_div(s3, total) * 100, 1),
                 data.get("scope3_sources_summary", "Purchased goods, travel, commuting")],
                ["TOTAL", round(total, 2), 100.0, ""],
            ],
        }

        top_sources_sheet = {
            "name": "Top Sources",
            "headers": ["Rank", "Source", "Scope", "Emissions (tCO2e)", "% of Total", "Data Quality"],
            "rows": [],
        }
        for idx, src in enumerate(data.get("top_sources", [])[:3], 1):
            top_sources_sheet["rows"].append([
                idx,
                src.get("name", ""),
                src.get("scope", ""),
                src.get("emissions_tco2e", 0),
                round(_safe_div(src.get("emissions_tco2e", 0), total) * 100, 1),
                src.get("data_quality", ""),
            ])

        peer_sheet = {
            "name": "Peer Comparison",
            "headers": ["Metric", "Your Value", "Sector Average", "Sector Best", "Percentile"],
            "rows": [
                ["Total Emissions (tCO2e)", round(total, 2),
                 data.get("sector_avg_tco2e", 0),
                 data.get("best_in_class_tco2e", 0),
                 data.get("percentile_rank", 50)],
                ["Intensity (tCO2e/FTE)", round(intensity, 2),
                 data.get("sector_avg_intensity", 0),
                 data.get("best_in_class_intensity", 0),
                 data.get("intensity_percentile", 50)],
            ],
        }

        next_steps_sheet = {
            "name": "Next Steps",
            "headers": ["Priority", "Action", "Impact", "Difficulty", "Timeline"],
            "rows": [],
        }
        for idx, step in enumerate(data.get("next_steps", []), 1):
            next_steps_sheet["rows"].append([
                idx,
                step.get("action", ""),
                step.get("impact", ""),
                step.get("difficulty", ""),
                step.get("timeline", ""),
            ])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sme_baseline_{data.get('org_name', 'org').replace(' ', '_')}_{data.get('reporting_year', '')}.xlsx",
            "worksheets": [
                summary_sheet,
                scope_sheet,
                top_sources_sheet,
                peer_sheet,
                next_steps_sheet,
            ],
            "chart_definitions": [
                {
                    "type": "bar",
                    "title": "Emissions by Scope",
                    "worksheet": "Scope Breakdown",
                    "data_range": "B2:B4",
                    "labels_range": "A2:A4",
                    "colors": [_ACCENT, "#66bb6a", "#a5d6a7"],
                },
                {
                    "type": "bar",
                    "title": "Peer Comparison",
                    "worksheet": "Peer Comparison",
                    "data_range": "B2:D2",
                    "labels_range": "B1:D1",
                    "colors": [_PRIMARY, _SECONDARY, _ACCENT],
                },
            ],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Your Company")
        year = data.get("reporting_year", "")
        return (
            f"# Emissions Baseline Dashboard\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Sector:** {data.get('sector', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)
        revenue = float(data.get("revenue", 0))
        currency = data.get("currency", "GBP")
        rev_intensity = _safe_div(total, revenue) * 1_000_000 if revenue > 0 else 0

        return (
            f"## Your Emissions at a Glance\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| **Total Emissions** | **{_dec_comma(total)} tCO2e** |\n"
            f"| Scope 1 (Direct) | {_dec_comma(s1)} tCO2e ({_pct(_safe_div(s1, total) * 100)}) |\n"
            f"| Scope 2 (Electricity) | {_dec_comma(s2)} tCO2e ({_pct(_safe_div(s2, total) * 100)}) |\n"
            f"| Scope 3 (Value Chain) | {_dec_comma(s3)} tCO2e ({_pct(_safe_div(s3, total) * 100)}) |\n"
            f"| **Per Employee** | **{_dec(intensity)} tCO2e / FTE** |\n"
            f"| Per {currency}1M Revenue | {_dec(rev_intensity)} tCO2e |\n"
            f"| Employees | {_dec_comma(employees)} |"
        )

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        lines = [
            "## Scope Breakdown\n",
            "| Scope | Emissions | % | What It Covers |",
            "|-------|----------:|---:|----------------|",
            f"| Scope 1 | {_dec_comma(s1)} tCO2e | {_pct(_safe_div(s1, total) * 100)} | Gas boilers, company vehicles, refrigerants |",
            f"| Scope 2 | {_dec_comma(s2)} tCO2e | {_pct(_safe_div(s2, total) * 100)} | Purchased electricity and heat |",
            f"| Scope 3 | {_dec_comma(s3)} tCO2e | {_pct(_safe_div(s3, total) * 100)} | Bought goods, travel, staff commuting |",
            f"| **Total** | **{_dec_comma(total)} tCO2e** | **100%** | |",
        ]
        return "\n".join(lines)

    def _md_scope_bars(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        mx = max(s1, s2, s3, 1)

        lines = [
            "### Visual Breakdown\n",
            f"```",
            f"Scope 1  [{_ascii_bar(s1, mx, 30)}] {_dec_comma(s1)} tCO2e",
            f"Scope 2  [{_ascii_bar(s2, mx, 30)}] {_dec_comma(s2)} tCO2e",
            f"Scope 3  [{_ascii_bar(s3, mx, 30)}] {_dec_comma(s3)} tCO2e",
            f"```",
        ]
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        sector = data.get("sector", "your sector")
        avg = float(data.get("sector_avg_tco2e", 0))
        median = float(data.get("sector_median_tco2e", 0))
        best = float(data.get("best_in_class_tco2e", 0))
        pctile = int(data.get("percentile_rank", 50))

        position = "below average (good!)" if total < avg else "above average"
        quartile = "Top 25%" if pctile >= 75 else ("Top 50%" if pctile >= 50 else "Bottom 50%")

        lines = [
            f"## How You Compare ({sector})\n",
            f"| Benchmark | Emissions | Your Position |",
            f"|-----------|----------:|:-------------:|",
            f"| Your Emissions | {_dec_comma(total)} tCO2e | - |",
            f"| Sector Average | {_dec_comma(avg)} tCO2e | {position} |",
            f"| Sector Median | {_dec_comma(median)} tCO2e | |",
            f"| Best in Class | {_dec_comma(best)} tCO2e | |",
            f"",
            f"**Your Percentile:** {pctile}th ({quartile})",
            f"",
            f"```",
            f"Best          Median        Average       Worst",
            f"  |-------------|-------------|-------------|",
        ]
        # Position marker
        marker_pos = int(round(pctile / 100 * 50))
        marker_pos = max(2, min(50, marker_pos))
        ruler = list("  " + "." * 51)
        ruler[marker_pos + 2] = "^"
        lines.append("".join(ruler))
        lines.append(f"{'':>{marker_pos + 2}}YOU")
        lines.append("```")

        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        score = float(data.get("data_quality_score", 0))
        tier = _dq_tier(score)
        details = data.get("data_quality_details", [])

        lines = [
            f"## Data Quality: {tier['label']} {tier['emoji']}\n",
            f"**Score:** {_dec(score, 0)}/100\n",
        ]
        if details:
            lines.append("| Data Source | Quality | Notes |")
            lines.append("|------------|:-------:|-------|")
            for d in details:
                lines.append(
                    f"| {d.get('source', '')} "
                    f"| {d.get('quality', '')} "
                    f"| {d.get('notes', '')} |"
                )
        lines.append("")
        if score < 50:
            lines.append("> **Tip:** Improve your data quality by using utility bills "
                         "instead of estimates. Connect your accounting software for automatic data capture.")
        elif score < 80:
            lines.append("> **Tip:** You have good foundations. Consider getting primary data "
                         "from your top suppliers to move to Gold quality.")
        else:
            lines.append("> **Well done!** Your data quality is excellent. This baseline "
                         "is suitable for third-party verification.")

        return "\n".join(lines)

    def _md_top_sources(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        sources = data.get("top_sources", [])[:3]

        lines = [
            "## Your Top 3 Emission Sources\n",
            "| Rank | Source | Scope | Emissions | % of Total |",
            "|:----:|--------|-------|----------:|-----------:|",
        ]
        for idx, src in enumerate(sources, 1):
            em = float(src.get("emissions_tco2e", 0))
            lines.append(
                f"| {idx} | {src.get('name', '')} "
                f"| {src.get('scope', '')} "
                f"| {_dec_comma(em)} tCO2e "
                f"| {_pct(_safe_div(em, total) * 100)} |"
            )
        if not sources:
            lines.append("| - | No sources identified yet | - | - | - |")

        return "\n".join(lines)

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        steps = data.get("next_steps", [])
        if not steps:
            steps = [
                {"action": "Review your energy bills to improve data quality",
                 "impact": "Better accuracy", "timeline": "This month"},
                {"action": "Switch to a renewable electricity tariff",
                 "impact": "Reduce Scope 2 to near-zero", "timeline": "Next quarter"},
                {"action": "Survey your top 5 suppliers on their emissions",
                 "impact": "Improve Scope 3 data", "timeline": "Within 6 months"},
            ]

        lines = ["## Recommended Next Steps\n"]
        for idx, step in enumerate(steps, 1):
            lines.append(
                f"**{idx}. {step.get('action', '')}**  \n"
                f"   Impact: {step.get('impact', '')} | "
                f"Timeline: {step.get('timeline', '')}  "
            )
            if step.get("difficulty"):
                lines.append(f"   Difficulty: {step.get('difficulty', '')}")
            lines.append("")

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Simplified GHG baseline report for SMEs. "
            f"Methodology: GHG Protocol SME Guide.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            f"padding:20px;background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:32px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;"
            f"font-size:1.8em;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};"
            f"padding-left:12px;font-size:1.3em;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.75em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}}"
            f".card-value{{font-size:1.6em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.7em;color:#689f38;}}"
            f".bar-container{{background:#e0e0e0;border-radius:6px;overflow:hidden;height:28px;"
            f"margin:4px 0;position:relative;}}"
            f".bar-fill{{height:100%;border-radius:6px;display:flex;align-items:center;"
            f"padding-left:8px;color:#fff;font-weight:600;font-size:0.8em;"
            f"transition:width 0.3s ease;}}"
            f".bar-s1{{background:{_ACCENT};}}"
            f".bar-s2{{background:#66bb6a;}}"
            f".bar-s3{{background:#a5d6a7;color:{_PRIMARY};}}"
            f".badge{{display:inline-block;padding:6px 16px;border-radius:20px;"
            f"font-weight:700;font-size:1.1em;}}"
            f".badge-gold{{background:#fff9c4;color:#f9a825;border:2px solid #ffd700;}}"
            f".badge-silver{{background:#f5f5f5;color:#757575;border:2px solid #c0c0c0;}}"
            f".badge-bronze{{background:#fff3e0;color:#e65100;border:2px solid #cd7f32;}}"
            f".peer-marker{{position:relative;height:40px;background:linear-gradient(to right,"
            f"{_ACCENT},#fdd835,#ff7043);border-radius:6px;margin:12px 0;}}"
            f".peer-you{{position:absolute;top:-8px;transform:translateX(-50%);"
            f"background:{_PRIMARY};color:#fff;padding:2px 8px;border-radius:4px;"
            f"font-size:0.75em;font-weight:600;}}"
            f".step-card{{background:{_LIGHTER};border-left:4px solid {_ACCENT};"
            f"padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;}}"
            f".step-num{{display:inline-block;background:{_PRIMARY};color:#fff;width:28px;"
            f"height:28px;border-radius:50%;text-align:center;line-height:28px;"
            f"font-weight:700;font-size:0.85em;margin-right:8px;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.summary-cards{{grid-template-columns:1fr 1fr;}}"
            f".report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Your Company")
        return (
            f'<h1>Emissions Baseline Dashboard</h1>\n'
            f'<p><strong>{org}</strong> | '
            f'{data.get("reporting_year", "")} | '
            f'{data.get("sector", "")} | '
            f'Generated: {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)

        return (
            f'<h2>Your Emissions at a Glance</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Emissions</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Per Employee</div>'
            f'<div class="card-value">{_dec(intensity)}</div>'
            f'<div class="card-unit">tCO2e / FTE</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(s1)}</div>'
            f'<div class="card-unit">tCO2e ({_pct(_safe_div(s1, total) * 100)})</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(s2)}</div>'
            f'<div class="card-unit">tCO2e ({_pct(_safe_div(s2, total) * 100)})</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(s3)}</div>'
            f'<div class="card-unit">tCO2e ({_pct(_safe_div(s3, total) * 100)})</div></div>\n'
            f'  <div class="card"><div class="card-label">Employees</div>'
            f'<div class="card-value">{_dec_comma(employees)}</div>'
            f'<div class="card-unit">FTE</div></div>\n'
            f'</div>'
        )

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        mx = max(s1, s2, s3, 1)

        bars = ""
        for scope, val, cls, label in [
            ("Scope 1", s1, "bar-s1", "Direct (gas, vehicles)"),
            ("Scope 2", s2, "bar-s2", "Electricity & heat"),
            ("Scope 3", s3, "bar-s3", "Value chain"),
        ]:
            pct = _safe_div(val, mx) * 100
            bars += (
                f'<p style="margin:4px 0 2px;font-weight:600;">{scope}: '
                f'{_dec_comma(val)} tCO2e ({_pct(_safe_div(val, total) * 100)})</p>\n'
                f'<div class="bar-container">'
                f'<div class="bar-fill {cls}" style="width:{max(pct, 3):.1f}%">'
                f'{label}</div></div>\n'
            )

        return (
            f'<h2>Scope Breakdown</h2>\n'
            f'{bars}'
            f'<p style="text-align:right;font-weight:700;color:{_PRIMARY};">'
            f'Total: {_dec_comma(total)} tCO2e</p>'
        )

    def _html_peer_comparison(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        avg = float(data.get("sector_avg_tco2e", 0))
        pctile = int(data.get("percentile_rank", 50))
        sector = data.get("sector", "Your Sector")
        position = "Below average (good!)" if total < avg else "Above average"

        return (
            f'<h2>How You Compare ({sector})</h2>\n'
            f'<div class="peer-marker">'
            f'<div class="peer-you" style="left:{pctile}%">YOU ({pctile}th %ile)</div>'
            f'</div>\n'
            f'<p style="display:flex;justify-content:space-between;font-size:0.8em;">'
            f'<span>Best in class</span><span>Sector average</span><span>Highest emitters</span></p>\n'
            f'<table>\n'
            f'<tr><th>Benchmark</th><th>Emissions</th><th>Position</th></tr>\n'
            f'<tr><td>Your Emissions</td><td>{_dec_comma(total)} tCO2e</td><td>{position}</td></tr>\n'
            f'<tr><td>Sector Average</td><td>{_dec_comma(avg)} tCO2e</td><td>-</td></tr>\n'
            f'<tr><td>Best in Class</td><td>{_dec_comma(data.get("best_in_class_tco2e", 0))} tCO2e</td>'
            f'<td>-</td></tr>\n'
            f'</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        score = float(data.get("data_quality_score", 0))
        tier = _dq_tier(score)
        badge_cls = f"badge-{tier['label'].lower()}"
        details = data.get("data_quality_details", [])

        rows = ""
        for d in details:
            rows += (
                f'<tr><td>{d.get("source", "")}</td>'
                f'<td>{d.get("quality", "")}</td>'
                f'<td>{d.get("notes", "")}</td></tr>\n'
            )

        tip = ""
        if score < 50:
            tip = "Improve by using utility bills instead of estimates. Connect your accounting software."
        elif score < 80:
            tip = "Good foundations. Get primary data from top suppliers to reach Gold."
        else:
            tip = "Excellent! Suitable for third-party verification."

        return (
            f'<h2>Data Quality</h2>\n'
            f'<p><span class="badge {badge_cls}">{tier["label"]}</span> '
            f'Score: {_dec(score, 0)}/100</p>\n'
            f'{"<table><tr><th>Source</th><th>Quality</th><th>Notes</th></tr>" + rows + "</table>" if rows else ""}\n'
            f'<p style="background:{_LIGHTER};padding:12px;border-radius:6px;">'
            f'<strong>Tip:</strong> {tip}</p>'
        )

    def _html_top_sources(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        sources = data.get("top_sources", [])[:3]

        rows = ""
        for idx, src in enumerate(sources, 1):
            em = float(src.get("emissions_tco2e", 0))
            rows += (
                f'<tr><td>{idx}</td><td>{src.get("name", "")}</td>'
                f'<td>{src.get("scope", "")}</td>'
                f'<td>{_dec_comma(em)} tCO2e</td>'
                f'<td>{_pct(_safe_div(em, total) * 100)}</td></tr>\n'
            )

        return (
            f'<h2>Top 3 Emission Sources</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Source</th><th>Scope</th>'
            f'<th>Emissions</th><th>% of Total</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_next_steps(self, data: Dict[str, Any]) -> str:
        steps = data.get("next_steps", [])
        if not steps:
            steps = [
                {"action": "Review your energy bills to improve data quality",
                 "impact": "Better accuracy", "timeline": "This month"},
                {"action": "Switch to a renewable electricity tariff",
                 "impact": "Reduce Scope 2 to near-zero", "timeline": "Next quarter"},
                {"action": "Survey your top 5 suppliers on their emissions",
                 "impact": "Improve Scope 3 data", "timeline": "Within 6 months"},
            ]

        cards = ""
        for idx, step in enumerate(steps, 1):
            cards += (
                f'<div class="step-card">'
                f'<span class="step-num">{idx}</span>'
                f'<strong>{step.get("action", "")}</strong><br>'
                f'<span style="font-size:0.85em;color:#558b2f;">'
                f'Impact: {step.get("impact", "")} | Timeline: {step.get("timeline", "")}'
                f'</span></div>\n'
            )

        return f'<h2>Recommended Next Steps</h2>\n{cards}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Simplified GHG baseline report for SMEs | GHG Protocol SME Guide'
            f'</div>'
        )
