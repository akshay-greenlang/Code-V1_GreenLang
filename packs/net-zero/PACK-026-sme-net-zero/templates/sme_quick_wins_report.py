# -*- coding: utf-8 -*-
"""
SMEQuickWinsReportTemplate - Top quick wins ranked by ROI for PACK-026.

Renders a prioritised list of 5-10 quick emission reduction actions for SMEs,
each showing emissions reduction, cost, savings, payback, IRR, difficulty,
and dependencies. Includes a 6-24 month Gantt-style timeline and 5-year
total investment/savings summary.

Sections:
    1. Quick Wins Summary (top actions ranked by ROI)
    2. Detailed Action Cards
    3. Implementation Timeline (Gantt chart)
    4. Financial Summary (investment + 5-year savings)
    5. Getting Started Checklist

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_quick_wins_report"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Difficulty labels
# ---------------------------------------------------------------------------
_DIFFICULTY_MAP = {
    1: {"label": "Very Easy", "stars": "*", "color": "#4caf50"},
    2: {"label": "Easy", "stars": "**", "color": "#8bc34a"},
    3: {"label": "Moderate", "stars": "***", "color": "#ffc107"},
    4: {"label": "Hard", "stars": "****", "color": "#ff9800"},
    5: {"label": "Very Hard", "stars": "*****", "color": "#f44336"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default


def _difficulty_info(level: int) -> Dict[str, Any]:
    return _DIFFICULTY_MAP.get(min(max(level, 1), 5), _DIFFICULTY_MAP[3])


def _gantt_bar(start_month: int, duration_months: int, total_months: int = 24,
               width: int = 24, char: str = "=") -> str:
    """Render a simple ASCII Gantt bar."""
    if total_months <= 0:
        return ""
    scale = width / total_months
    offset = int(round(start_month * scale))
    length = max(1, int(round(duration_months * scale)))
    offset = max(0, min(width - 1, offset))
    length = min(length, width - offset)
    return "." * offset + char * length + "." * (width - offset - length)


# ===========================================================================
# Template Class
# ===========================================================================

class SMEQuickWinsReportTemplate:
    """
    SME quick wins report template: top 5-10 actions ranked by ROI.

    Renders a prioritised action plan showing emissions reduction, cost,
    savings, payback period, IRR, difficulty, prerequisites, and a Gantt
    timeline across Markdown, HTML, JSON, and Excel formats.
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
        """Render the quick wins report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary_table(data),
            self._md_action_cards(data),
            self._md_gantt_timeline(data),
            self._md_financial_summary(data),
            self._md_getting_started(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the quick wins report as HTML with inline CSS."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_summary_table(data),
            self._html_action_cards(data),
            self._html_gantt_timeline(data),
            self._html_financial_summary(data),
            self._html_getting_started(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Quick Wins Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the quick wins report as structured JSON."""
        self.generated_at = _utcnow()
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        total_investment = sum(float(a.get("cost", 0)) for a in actions)
        total_annual_savings = sum(float(a.get("annual_savings", 0)) for a in actions)
        total_5yr_savings = total_annual_savings * 5
        total_reduction = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
        total_reduction_pct = float(data.get("total_reduction_pct", 0))

        processed_actions = []
        for idx, a in enumerate(actions, 1):
            diff_info = _difficulty_info(int(a.get("difficulty", 3)))
            processed_actions.append({
                "rank": idx,
                "name": a.get("name", ""),
                "category": a.get("category", ""),
                "scope_impacted": a.get("scope_impacted", ""),
                "reduction_tco2e": round(float(a.get("reduction_tco2e", 0)), 2),
                "reduction_pct": round(float(a.get("reduction_pct", 0)), 1),
                "cost": round(float(a.get("cost", 0)), 2),
                "annual_savings": round(float(a.get("annual_savings", 0)), 2),
                "payback_months": int(a.get("payback_months", 0)),
                "irr_pct": round(float(a.get("irr_pct", 0)), 1),
                "difficulty": int(a.get("difficulty", 3)),
                "difficulty_label": diff_info["label"],
                "start_month": int(a.get("start_month", 0)),
                "duration_months": int(a.get("duration_months", 3)),
                "prerequisites": a.get("prerequisites", []),
                "dependencies": a.get("dependencies", []),
            })

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "baseline_tco2e": data.get("baseline_tco2e", 0),
                "currency": currency,
            },
            "quick_wins": processed_actions,
            "financial_summary": {
                "total_investment": round(total_investment, 2),
                "total_annual_savings": round(total_annual_savings, 2),
                "total_5yr_savings": round(total_5yr_savings, 2),
                "net_5yr_benefit": round(total_5yr_savings - total_investment, 2),
                "total_reduction_tco2e": round(total_reduction, 2),
                "total_reduction_pct": total_reduction_pct,
                "average_payback_months": round(
                    _safe_div(
                        sum(int(a.get("payback_months", 0)) for a in actions),
                        len(actions)
                    ), 1
                ) if actions else 0,
                "currency": currency,
            },
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure with worksheet definitions."""
        self.generated_at = _utcnow()
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        total_investment = sum(float(a.get("cost", 0)) for a in actions)
        total_annual_savings = sum(float(a.get("annual_savings", 0)) for a in actions)
        total_reduction = sum(float(a.get("reduction_tco2e", 0)) for a in actions)

        ranked_sheet = {
            "name": "Quick Wins Ranked",
            "headers": [
                "Rank", "Action", "Category", "Scope",
                "Reduction (tCO2e)", "Reduction (%)",
                f"Cost ({currency})", f"Annual Savings ({currency})",
                "Payback (months)", "IRR (%)", "Difficulty (1-5)",
            ],
            "rows": [],
        }
        for idx, a in enumerate(actions, 1):
            ranked_sheet["rows"].append([
                idx,
                a.get("name", ""),
                a.get("category", ""),
                a.get("scope_impacted", ""),
                round(float(a.get("reduction_tco2e", 0)), 2),
                round(float(a.get("reduction_pct", 0)), 1),
                round(float(a.get("cost", 0)), 2),
                round(float(a.get("annual_savings", 0)), 2),
                int(a.get("payback_months", 0)),
                round(float(a.get("irr_pct", 0)), 1),
                int(a.get("difficulty", 3)),
            ])
        ranked_sheet["rows"].append([
            "", "TOTAL", "", "",
            round(total_reduction, 2), "",
            round(total_investment, 2),
            round(total_annual_savings, 2),
            "", "", "",
        ])

        timeline_sheet = {
            "name": "Timeline",
            "headers": ["Action", "Start Month", "Duration (months)", "End Month"],
            "rows": [],
        }
        for a in actions:
            start = int(a.get("start_month", 0))
            dur = int(a.get("duration_months", 3))
            timeline_sheet["rows"].append([
                a.get("name", ""),
                start,
                dur,
                start + dur,
            ])

        financial_sheet = {
            "name": "Financial Summary",
            "headers": ["Metric", "Value", "Unit"],
            "rows": [
                ["Total Investment", round(total_investment, 2), currency],
                ["Annual Savings", round(total_annual_savings, 2), currency],
                ["5-Year Savings", round(total_annual_savings * 5, 2), currency],
                ["Net 5-Year Benefit", round(total_annual_savings * 5 - total_investment, 2), currency],
                ["Total Emission Reduction", round(total_reduction, 2), "tCO2e"],
                ["Actions Count", len(actions), ""],
            ],
        }

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sme_quick_wins_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [ranked_sheet, timeline_sheet, financial_sheet],
            "chart_definitions": [
                {
                    "type": "bar",
                    "title": "Reduction by Action (tCO2e)",
                    "worksheet": "Quick Wins Ranked",
                    "data_range": "E2:E" + str(len(actions) + 1),
                    "labels_range": "B2:B" + str(len(actions) + 1),
                    "colors": [_ACCENT],
                },
                {
                    "type": "horizontal_bar",
                    "title": "Implementation Timeline",
                    "worksheet": "Timeline",
                    "data_range": "B2:C" + str(len(actions) + 1),
                    "labels_range": "A2:A" + str(len(actions) + 1),
                    "colors": [_SECONDARY, _ACCENT],
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
        return (
            f"# Quick Wins Action Plan\n\n"
            f"**Organization:** {org}  \n"
            f"**Baseline Emissions:** {_dec_comma(data.get('baseline_tco2e', 0))} tCO2e  \n"
            f"**Currency:** {data.get('currency', 'GBP')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_summary_table(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        lines = [
            "## Quick Wins Summary (Ranked by ROI)\n",
            f"| # | Action | Scope | Reduction | Cost ({currency}) | Savings/yr | Payback | Difficulty |",
            f"|--:|--------|-------|----------:|----------:|----------:|--------:|:----------:|",
        ]
        for idx, a in enumerate(actions, 1):
            diff = _difficulty_info(int(a.get("difficulty", 3)))
            lines.append(
                f"| {idx} "
                f"| {a.get('name', '')} "
                f"| {a.get('scope_impacted', '')} "
                f"| {_dec_comma(a.get('reduction_tco2e', 0))} tCO2e "
                f"| {_dec_comma(a.get('cost', 0))} "
                f"| {_dec_comma(a.get('annual_savings', 0))} "
                f"| {a.get('payback_months', 0)} mo "
                f"| {diff['stars']} |"
            )
        return "\n".join(lines)

    def _md_action_cards(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        lines = ["## Detailed Action Cards\n"]
        for idx, a in enumerate(actions, 1):
            diff = _difficulty_info(int(a.get("difficulty", 3)))
            prereqs = a.get("prerequisites", [])
            deps = a.get("dependencies", [])

            lines.append(f"### {idx}. {a.get('name', '')}")
            lines.append(f"**Category:** {a.get('category', '')} | "
                         f"**Scope:** {a.get('scope_impacted', '')} | "
                         f"**Difficulty:** {diff['label']} ({diff['stars']})\n")
            lines.append(
                f"| Metric | Value |\n"
                f"|--------|------:|\n"
                f"| Emissions Reduction | {_dec_comma(a.get('reduction_tco2e', 0))} tCO2e "
                f"({_pct(a.get('reduction_pct', 0))}) |\n"
                f"| Implementation Cost | {currency} {_dec_comma(a.get('cost', 0))} |\n"
                f"| Annual Savings | {currency} {_dec_comma(a.get('annual_savings', 0))} |\n"
                f"| Payback Period | {a.get('payback_months', 0)} months |\n"
                f"| IRR | {_pct(a.get('irr_pct', 0))} |\n"
                f"| Start Month | Month {a.get('start_month', 0)} |\n"
                f"| Duration | {a.get('duration_months', 3)} months |"
            )
            if prereqs:
                lines.append(f"\n**Prerequisites:** {', '.join(prereqs)}")
            if deps:
                lines.append(f"**Dependencies:** {', '.join(deps)}")
            lines.append("")

        return "\n".join(lines)

    def _md_gantt_timeline(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        total_months = int(data.get("timeline_months", 24))

        lines = [
            f"## Implementation Timeline ({total_months} months)\n",
            f"```",
            f"{'Action':<30} {'Month':>5} {'Timeline':<{total_months + 4}}",
            f"{'-' * 30} {'-' * 5} {'-' * (total_months + 4)}",
        ]

        for a in actions:
            name = a.get("name", "")[:28]
            start = int(a.get("start_month", 0))
            dur = int(a.get("duration_months", 3))
            bar = _gantt_bar(start, dur, total_months, total_months)
            lines.append(f"{name:<30} {start:>5} [{bar}]")

        # Month scale
        scale = ""
        for m in range(0, total_months + 1, 6):
            scale += f"{m:<6}"
        lines.append(f"{'':>36} {scale}")
        lines.append("```")

        return "\n".join(lines)

    def _md_financial_summary(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        total_investment = sum(float(a.get("cost", 0)) for a in actions)
        total_annual_savings = sum(float(a.get("annual_savings", 0)) for a in actions)
        total_reduction = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
        total_5yr = total_annual_savings * 5
        net_benefit = total_5yr - total_investment

        lines = [
            "## Financial Summary\n",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| Total Investment | {currency} {_dec_comma(total_investment)} |",
            f"| Annual Savings | {currency} {_dec_comma(total_annual_savings)} |",
            f"| 5-Year Total Savings | {currency} {_dec_comma(total_5yr)} |",
            f"| **Net 5-Year Benefit** | **{currency} {_dec_comma(net_benefit)}** |",
            f"| Total Emission Reduction | {_dec_comma(total_reduction)} tCO2e |",
            f"| Reduction (% of baseline) | {_pct(data.get('total_reduction_pct', 0))} |",
            f"| Number of Actions | {len(actions)} |",
            f"",
            f"**Return on Investment:** For every {currency}1 invested, you save "
            f"{currency}{_dec(_safe_div(total_5yr, total_investment))} over 5 years.",
        ]

        return "\n".join(lines)

    def _md_getting_started(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        first_action = actions[0].get("name", "your first action") if actions else "your first action"

        lines = [
            "## Getting Started\n",
            f"1. **Start with #{1}:** {first_action} - the quickest ROI",
            f"2. **Set a budget:** Allocate funds from your savings to reinvest",
            f"3. **Track monthly:** Monitor energy bills to verify savings",
            f"4. **Celebrate wins:** Share progress with your team",
            f"5. **Scale up:** Use savings to fund the next action\n",
            f"> **Tip:** You don't need to do everything at once. Start with the "
            f"easiest, highest-ROI action and build momentum.",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Quick wins prioritised by return on investment for SMEs.*"
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
            f"h3{{color:#388e3c;margin-top:20px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".action-card{{background:{_LIGHTER};border:1px solid {_CARD_BG};"
            f"border-left:5px solid {_ACCENT};border-radius:0 10px 10px 0;"
            f"padding:16px;margin:12px 0;}}"
            f".action-header{{display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:8px;}}"
            f".action-title{{font-size:1.1em;font-weight:700;color:{_PRIMARY};}}"
            f".difficulty{{display:inline-block;padding:3px 10px;border-radius:12px;"
            f"font-size:0.8em;font-weight:600;color:#fff;}}"
            f".action-metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));"
            f"gap:8px;margin-top:10px;}}"
            f".metric{{text-align:center;padding:8px;background:#fff;border-radius:6px;}}"
            f".metric-label{{font-size:0.7em;color:#689f38;text-transform:uppercase;}}"
            f".metric-value{{font-size:1.1em;font-weight:700;color:{_PRIMARY};}}"
            f".gantt-container{{overflow-x:auto;margin:16px 0;}}"
            f".gantt-row{{display:flex;align-items:center;margin:4px 0;}}"
            f".gantt-label{{width:200px;min-width:200px;font-size:0.85em;padding-right:8px;"
            f"text-align:right;font-weight:600;color:{_PRIMARY};}}"
            f".gantt-track{{flex:1;height:24px;background:#f0f0f0;border-radius:4px;"
            f"position:relative;overflow:hidden;}}"
            f".gantt-bar{{position:absolute;height:100%;border-radius:4px;"
            f"background:{_ACCENT};display:flex;align-items:center;padding-left:6px;"
            f"color:#fff;font-size:0.7em;font-weight:600;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.75em;color:#558b2f;text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.7em;color:#689f38;}}"
            f".checklist{{list-style:none;padding:0;}}"
            f".checklist li{{padding:6px 0;border-bottom:1px solid {_CARD_BG};}}"
            f".checklist li::before{{content:'[ ] ';font-weight:700;color:{_ACCENT};}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.action-metrics{{grid-template-columns:1fr 1fr;}}"
            f".gantt-label{{width:120px;min-width:120px;}}.report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Quick Wins Action Plan</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'Baseline: {_dec_comma(data.get("baseline_tco2e", 0))} tCO2e | '
            f'Generated: {ts}</p>'
        )

    def _html_summary_table(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        rows = ""
        for idx, a in enumerate(actions, 1):
            diff = _difficulty_info(int(a.get("difficulty", 3)))
            rows += (
                f'<tr><td>{idx}</td>'
                f'<td><strong>{a.get("name", "")}</strong></td>'
                f'<td>{a.get("scope_impacted", "")}</td>'
                f'<td>{_dec_comma(a.get("reduction_tco2e", 0))} tCO2e</td>'
                f'<td>{currency} {_dec_comma(a.get("cost", 0))}</td>'
                f'<td>{currency} {_dec_comma(a.get("annual_savings", 0))}</td>'
                f'<td>{a.get("payback_months", 0)} mo</td>'
                f'<td><span class="difficulty" style="background:{diff["color"]}">'
                f'{diff["label"]}</span></td></tr>\n'
            )

        return (
            f'<h2>Quick Wins Summary (Ranked by ROI)</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Scope</th><th>Reduction</th>'
            f'<th>Cost</th><th>Savings/yr</th><th>Payback</th><th>Difficulty</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_action_cards(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        cards = ""
        for idx, a in enumerate(actions, 1):
            diff = _difficulty_info(int(a.get("difficulty", 3)))
            prereqs = a.get("prerequisites", [])

            prereq_html = ""
            if prereqs:
                prereq_html = (
                    f'<p style="font-size:0.85em;margin-top:8px;">'
                    f'<strong>Prerequisites:</strong> {", ".join(prereqs)}</p>'
                )

            cards += (
                f'<div class="action-card">\n'
                f'  <div class="action-header">'
                f'    <span class="action-title">{idx}. {a.get("name", "")}</span>'
                f'    <span class="difficulty" style="background:{diff["color"]}">'
                f'{diff["label"]}</span>'
                f'  </div>\n'
                f'  <p style="font-size:0.85em;color:#689f38;">'
                f'{a.get("category", "")} | {a.get("scope_impacted", "")}</p>\n'
                f'  <div class="action-metrics">\n'
                f'    <div class="metric"><div class="metric-label">Reduction</div>'
                f'<div class="metric-value">{_dec_comma(a.get("reduction_tco2e", 0))}</div>'
                f'<div class="metric-label">tCO2e ({_pct(a.get("reduction_pct", 0))})</div></div>\n'
                f'    <div class="metric"><div class="metric-label">Cost</div>'
                f'<div class="metric-value">{currency} {_dec_comma(a.get("cost", 0))}</div></div>\n'
                f'    <div class="metric"><div class="metric-label">Savings/yr</div>'
                f'<div class="metric-value">{currency} {_dec_comma(a.get("annual_savings", 0))}</div></div>\n'
                f'    <div class="metric"><div class="metric-label">Payback</div>'
                f'<div class="metric-value">{a.get("payback_months", 0)} mo</div></div>\n'
                f'    <div class="metric"><div class="metric-label">IRR</div>'
                f'<div class="metric-value">{_pct(a.get("irr_pct", 0))}</div></div>\n'
                f'  </div>\n'
                f'  {prereq_html}\n'
                f'</div>\n'
            )

        return f'<h2>Detailed Action Cards</h2>\n{cards}'

    def _html_gantt_timeline(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        total_months = int(data.get("timeline_months", 24))

        rows = ""
        for a in actions:
            start = int(a.get("start_month", 0))
            dur = int(a.get("duration_months", 3))
            left_pct = _safe_div(start, total_months) * 100
            width_pct = max(_safe_div(dur, total_months) * 100, 4)
            name = a.get("name", "")[:25]

            rows += (
                f'<div class="gantt-row">'
                f'<div class="gantt-label">{name}</div>'
                f'<div class="gantt-track">'
                f'<div class="gantt-bar" style="left:{left_pct:.1f}%;width:{width_pct:.1f}%">'
                f'M{start}-{start + dur}</div></div></div>\n'
            )

        # Month markers
        markers = '<div style="display:flex;margin-left:200px;font-size:0.7em;color:#9e9e9e;">'
        for m in range(0, total_months + 1, 6):
            markers += f'<span style="flex:1;">M{m}</span>'
        markers += '</div>'

        return (
            f'<h2>Implementation Timeline</h2>\n'
            f'<div class="gantt-container">\n{rows}{markers}\n</div>'
        )

    def _html_financial_summary(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        currency = data.get("currency", "GBP")

        total_investment = sum(float(a.get("cost", 0)) for a in actions)
        total_annual_savings = sum(float(a.get("annual_savings", 0)) for a in actions)
        total_5yr = total_annual_savings * 5
        net_benefit = total_5yr - total_investment
        total_reduction = sum(float(a.get("reduction_tco2e", 0)) for a in actions)

        return (
            f'<h2>Financial Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Investment</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_investment)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual Savings</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_annual_savings)}</div></div>\n'
            f'  <div class="card"><div class="card-label">5-Year Savings</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_5yr)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net 5yr Benefit</div>'
            f'<div class="card-value">{currency} {_dec_comma(net_benefit)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Emission Reduction</div>'
            f'<div class="card-value">{_dec_comma(total_reduction)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Actions</div>'
            f'<div class="card-value">{len(actions)}</div></div>\n'
            f'</div>'
        )

    def _html_getting_started(self, data: Dict[str, Any]) -> str:
        actions = data.get("quick_wins", [])
        first = actions[0].get("name", "your first action") if actions else "your first action"

        return (
            f'<h2>Getting Started</h2>\n'
            f'<ul class="checklist">\n'
            f'  <li>Start with #1: {first} - the quickest ROI</li>\n'
            f'  <li>Set a budget: allocate funds from savings to reinvest</li>\n'
            f'  <li>Track monthly: monitor energy bills to verify savings</li>\n'
            f'  <li>Celebrate wins: share progress with your team</li>\n'
            f'  <li>Scale up: use savings to fund the next action</li>\n'
            f'</ul>\n'
            f'<p style="background:{_LIGHTER};padding:12px;border-radius:6px;">'
            f'<strong>Tip:</strong> Start with the easiest, highest-ROI action '
            f'and build momentum.</p>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Quick wins prioritised by ROI for SMEs'
            f'</div>'
        )
