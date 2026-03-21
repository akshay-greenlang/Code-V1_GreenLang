# -*- coding: utf-8 -*-
"""
SMERoadmapReportTemplate - 3-year decarbonization roadmap for PACK-026.

Renders a phased 3-year decarbonization roadmap for SMEs with Year 1
quick wins, Year 2 strategic actions, Year 3 long-term investments,
MACC curve data, budget allocation, milestones, and KPIs.

Sections:
    1. Roadmap Overview (3-year trajectory)
    2. Year 1: Quick Wins (LED, renewable PPA, waste)
    3. Year 2: Strategic Actions (HVAC, EV fleet, supplier engagement)
    4. Year 3: Long-Term Investments (solar PV, heat pumps, process)
    5. MACC Curve Visualization (cost per tCO2e)
    6. Budget Allocation Table
    7. Milestones & KPIs

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
_TEMPLATE_ID = "sme_roadmap_report"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

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


def _ascii_bar(value: float, max_value: float, width: int = 30, char: str = "#") -> str:
    if max_value <= 0:
        return ""
    filled = int(round(value / max_value * width))
    filled = max(0, min(width, filled))
    return char * filled + "." * (width - filled)


# ===========================================================================
# Template Class
# ===========================================================================

class SMERoadmapReportTemplate:
    """
    SME 3-year decarbonization roadmap template.

    Renders a phased roadmap with Year 1/2/3 actions, emissions
    trajectory, investment/savings, MACC curve, budget allocation,
    milestones, and KPIs across Markdown, HTML, JSON, and Excel formats.
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
        """Render the roadmap report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_trajectory(data),
            self._md_year_detail(data, 1),
            self._md_year_detail(data, 2),
            self._md_year_detail(data, 3),
            self._md_macc_curve(data),
            self._md_budget_allocation(data),
            self._md_milestones(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the roadmap report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_trajectory(data),
            self._html_year_detail(data, 1),
            self._html_year_detail(data, 2),
            self._html_year_detail(data, 3),
            self._html_macc_curve(data),
            self._html_budget_allocation(data),
            self._html_milestones(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME 3-Year Decarbonization Roadmap</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the roadmap report as structured JSON."""
        self.generated_at = _utcnow()
        baseline = float(data.get("baseline_tco2e", 0))
        currency = data.get("currency", "GBP")

        years = {}
        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            actions = yr_data.get("actions", [])
            reduction = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
            investment = sum(float(a.get("cost", 0)) for a in actions)
            savings = sum(float(a.get("annual_savings", 0)) for a in actions)

            years[f"year_{yr}"] = {
                "label": yr_data.get("label", f"Year {yr}"),
                "theme": yr_data.get("theme", ""),
                "actions": [
                    {
                        "name": a.get("name", ""),
                        "category": a.get("category", ""),
                        "reduction_tco2e": round(float(a.get("reduction_tco2e", 0)), 2),
                        "cost": round(float(a.get("cost", 0)), 2),
                        "annual_savings": round(float(a.get("annual_savings", 0)), 2),
                        "cost_per_tco2e": round(
                            _safe_div(float(a.get("cost", 0)), float(a.get("reduction_tco2e", 1))), 2
                        ),
                    }
                    for a in actions
                ],
                "total_reduction_tco2e": round(reduction, 2),
                "total_investment": round(investment, 2),
                "total_annual_savings": round(savings, 2),
                "remaining_tco2e": round(baseline - reduction, 2),
            }

        # Cumulative trajectory
        cumulative_reduction = 0.0
        trajectory = []
        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            yr_reduction = sum(float(a.get("reduction_tco2e", 0)) for a in yr_data.get("actions", []))
            cumulative_reduction += yr_reduction
            trajectory.append({
                "year": yr,
                "cumulative_reduction_tco2e": round(cumulative_reduction, 2),
                "remaining_tco2e": round(baseline - cumulative_reduction, 2),
                "reduction_pct": round(_safe_div(cumulative_reduction, baseline) * 100, 1),
            })

        # MACC data
        macc_entries = data.get("macc_data", [])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "baseline_tco2e": baseline,
                "currency": currency,
            },
            "years": years,
            "trajectory": trajectory,
            "macc_data": macc_entries,
            "milestones": data.get("milestones", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure."""
        self.generated_at = _utcnow()
        baseline = float(data.get("baseline_tco2e", 0))
        currency = data.get("currency", "GBP")

        # Overview sheet
        overview_sheet = {
            "name": "Overview",
            "headers": ["Year", "Theme", "Actions", "Reduction (tCO2e)",
                        f"Investment ({currency})", f"Savings/yr ({currency})",
                        "Remaining (tCO2e)", "Reduction (%)"],
            "rows": [],
        }
        cumulative = 0.0
        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            actions = yr_data.get("actions", [])
            red = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
            inv = sum(float(a.get("cost", 0)) for a in actions)
            sav = sum(float(a.get("annual_savings", 0)) for a in actions)
            cumulative += red
            overview_sheet["rows"].append([
                f"Year {yr}",
                yr_data.get("theme", ""),
                len(actions),
                round(red, 2),
                round(inv, 2),
                round(sav, 2),
                round(baseline - cumulative, 2),
                round(_safe_div(cumulative, baseline) * 100, 1),
            ])

        # Actions sheet (all years)
        actions_sheet = {
            "name": "All Actions",
            "headers": ["Year", "Action", "Category", "Reduction (tCO2e)",
                        f"Cost ({currency})", f"Savings/yr ({currency})",
                        f"Cost/tCO2e ({currency})"],
            "rows": [],
        }
        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            for a in yr_data.get("actions", []):
                red = float(a.get("reduction_tco2e", 0))
                cost = float(a.get("cost", 0))
                actions_sheet["rows"].append([
                    f"Year {yr}",
                    a.get("name", ""),
                    a.get("category", ""),
                    round(red, 2),
                    round(cost, 2),
                    round(float(a.get("annual_savings", 0)), 2),
                    round(_safe_div(cost, red), 2),
                ])

        # MACC sheet
        macc_sheet = {
            "name": "MACC Curve",
            "headers": ["Action", f"Cost/tCO2e ({currency})", "Reduction (tCO2e)", "Cumulative Reduction"],
            "rows": [],
        }
        macc_data = data.get("macc_data", [])
        cum_red = 0.0
        for m in sorted(macc_data, key=lambda x: float(x.get("cost_per_tco2e", 0))):
            red = float(m.get("reduction_tco2e", 0))
            cum_red += red
            macc_sheet["rows"].append([
                m.get("name", ""),
                round(float(m.get("cost_per_tco2e", 0)), 2),
                round(red, 2),
                round(cum_red, 2),
            ])

        # Milestones sheet
        milestones_sheet = {
            "name": "Milestones & KPIs",
            "headers": ["Year", "Quarter", "Milestone", "KPI", "Target"],
            "rows": [],
        }
        for ms in data.get("milestones", []):
            milestones_sheet["rows"].append([
                ms.get("year", ""),
                ms.get("quarter", ""),
                ms.get("milestone", ""),
                ms.get("kpi", ""),
                ms.get("target", ""),
            ])

        # Budget sheet
        budget_sheet = {
            "name": "Budget Allocation",
            "headers": ["Category", f"Year 1 ({currency})", f"Year 2 ({currency})",
                        f"Year 3 ({currency})", f"Total ({currency})"],
            "rows": [],
        }
        for cat in data.get("budget_categories", []):
            budget_sheet["rows"].append([
                cat.get("category", ""),
                round(float(cat.get("year1", 0)), 2),
                round(float(cat.get("year2", 0)), 2),
                round(float(cat.get("year3", 0)), 2),
                round(float(cat.get("year1", 0)) + float(cat.get("year2", 0))
                      + float(cat.get("year3", 0)), 2),
            ])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sme_roadmap_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [overview_sheet, actions_sheet, macc_sheet, milestones_sheet, budget_sheet],
            "chart_definitions": [
                {
                    "type": "line",
                    "title": "Emissions Reduction Trajectory",
                    "worksheet": "Overview",
                    "data_range": "G2:G4",
                    "labels_range": "A2:A4",
                    "colors": [_PRIMARY],
                },
                {
                    "type": "bar",
                    "title": "MACC Curve",
                    "worksheet": "MACC Curve",
                    "data_range": "B2:B" + str(len(macc_data) + 1),
                    "labels_range": "A2:A" + str(len(macc_data) + 1),
                    "colors": [_ACCENT],
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
        return (
            f"# 3-Year Decarbonization Roadmap\n\n"
            f"**Organization:** {data.get('org_name', 'Your Company')}  \n"
            f"**Baseline Emissions:** {_dec_comma(data.get('baseline_tco2e', 0))} tCO2e  \n"
            f"**Currency:** {data.get('currency', 'GBP')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        baseline = float(data.get("baseline_tco2e", 0))
        currency = data.get("currency", "GBP")

        lines = [
            "## Roadmap Overview\n",
            "| Year | Theme | Actions | Reduction | Investment | Savings/yr |",
            "|------|-------|--------:|----------:|-----------:|-----------:|",
        ]
        cumulative = 0.0
        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            actions = yr_data.get("actions", [])
            red = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
            inv = sum(float(a.get("cost", 0)) for a in actions)
            sav = sum(float(a.get("annual_savings", 0)) for a in actions)
            cumulative += red
            lines.append(
                f"| Year {yr} | {yr_data.get('theme', '')} "
                f"| {len(actions)} "
                f"| {_dec_comma(red)} tCO2e "
                f"| {currency} {_dec_comma(inv)} "
                f"| {currency} {_dec_comma(sav)} |"
            )

        total_red = cumulative
        lines.append(
            f"\n**Total 3-Year Reduction:** {_dec_comma(total_red)} tCO2e "
            f"({_pct(_safe_div(total_red, baseline) * 100)} of baseline)"
        )

        return "\n".join(lines)

    def _md_trajectory(self, data: Dict[str, Any]) -> str:
        baseline = float(data.get("baseline_tco2e", 0))

        lines = [
            "## Emissions Trajectory\n",
            "```",
            f"{'Year':<10} {'Emissions':>12} {'Reduction':>12} {'Chart':<32}",
            f"{'-' * 10} {'-' * 12} {'-' * 12} {'-' * 32}",
        ]

        cumulative = 0.0
        remaining = baseline
        for yr in [0, 1, 2, 3]:
            if yr == 0:
                label = "Baseline"
                bar = _ascii_bar(baseline, baseline, 30)
                lines.append(f"{label:<10} {_dec_comma(baseline):>12} {'':>12} [{bar}]")
            else:
                yr_data = data.get(f"year{yr}", {})
                yr_red = sum(float(a.get("reduction_tco2e", 0))
                             for a in yr_data.get("actions", []))
                cumulative += yr_red
                remaining = baseline - cumulative
                bar = _ascii_bar(remaining, baseline, 30)
                lines.append(
                    f"{'Year ' + str(yr):<10} "
                    f"{_dec_comma(remaining):>12} "
                    f"{'-' + _dec_comma(yr_red):>12} "
                    f"[{bar}]"
                )

        lines.append("```")
        return "\n".join(lines)

    def _md_year_detail(self, data: Dict[str, Any], year: int) -> str:
        yr_data = data.get(f"year{year}", {})
        theme = yr_data.get("theme", f"Year {year}")
        actions = yr_data.get("actions", [])
        currency = data.get("currency", "GBP")

        total_red = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
        total_inv = sum(float(a.get("cost", 0)) for a in actions)
        total_sav = sum(float(a.get("annual_savings", 0)) for a in actions)

        lines = [
            f"## Year {year}: {theme}\n",
            f"**Actions:** {len(actions)} | "
            f"**Reduction:** {_dec_comma(total_red)} tCO2e | "
            f"**Investment:** {currency} {_dec_comma(total_inv)} | "
            f"**Savings/yr:** {currency} {_dec_comma(total_sav)}\n",
            f"| Action | Category | Reduction | Cost | Savings/yr | Cost/tCO2e |",
            f"|--------|----------|----------:|-----:|-----------:|-----------:|",
        ]
        for a in actions:
            red = float(a.get("reduction_tco2e", 0))
            cost = float(a.get("cost", 0))
            cpt = _safe_div(cost, red)
            lines.append(
                f"| {a.get('name', '')} "
                f"| {a.get('category', '')} "
                f"| {_dec_comma(red)} tCO2e "
                f"| {currency} {_dec_comma(cost)} "
                f"| {currency} {_dec_comma(a.get('annual_savings', 0))} "
                f"| {currency} {_dec_comma(cpt)} |"
            )

        # KPIs for this year
        kpis = yr_data.get("kpis", [])
        if kpis:
            lines.append(f"\n**Year {year} KPIs:**")
            for kpi in kpis:
                lines.append(f"- {kpi.get('name', '')}: {kpi.get('target', '')}")

        return "\n".join(lines)

    def _md_macc_curve(self, data: Dict[str, Any]) -> str:
        macc_data = data.get("macc_data", [])
        currency = data.get("currency", "GBP")

        if not macc_data:
            return "## MACC Curve\n\n*No MACC data provided.*"

        sorted_macc = sorted(macc_data, key=lambda x: float(x.get("cost_per_tco2e", 0)))
        max_cost = max(abs(float(m.get("cost_per_tco2e", 0))) for m in sorted_macc) if sorted_macc else 1

        lines = [
            "## Marginal Abatement Cost Curve (MACC)\n",
            f"Actions sorted by cost per tCO2e - negative costs save money:\n",
            "```",
            f"{'Action':<25} {'Cost/tCO2e':>12} {'Reduction':>10} {'Chart':<20}",
            f"{'-' * 25} {'-' * 12} {'-' * 10} {'-' * 20}",
        ]
        for m in sorted_macc:
            cost = float(m.get("cost_per_tco2e", 0))
            red = float(m.get("reduction_tco2e", 0))
            name = m.get("name", "")[:23]
            # Use < for negative (savings), > for positive (costs)
            if cost < 0:
                bar_len = int(round(abs(cost) / max_cost * 10))
                bar = "<" * bar_len
            else:
                bar_len = int(round(cost / max_cost * 10))
                bar = ">" * bar_len
            lines.append(
                f"{name:<25} "
                f"{currency + ' ' + _dec_comma(cost):>12} "
                f"{_dec_comma(red):>10} "
                f"{bar}"
            )

        lines.append("```")
        lines.append(f"\n*< = saves money per tCO2e | > = costs money per tCO2e*")

        return "\n".join(lines)

    def _md_budget_allocation(self, data: Dict[str, Any]) -> str:
        categories = data.get("budget_categories", [])
        currency = data.get("currency", "GBP")

        if not categories:
            return ""

        lines = [
            "## Budget Allocation\n",
            f"| Category | Year 1 ({currency}) | Year 2 ({currency}) | Year 3 ({currency}) | Total ({currency}) |",
            f"|----------|----------:|----------:|----------:|-------:|",
        ]
        totals = [0.0, 0.0, 0.0]
        for cat in categories:
            y1 = float(cat.get("year1", 0))
            y2 = float(cat.get("year2", 0))
            y3 = float(cat.get("year3", 0))
            totals[0] += y1
            totals[1] += y2
            totals[2] += y3
            lines.append(
                f"| {cat.get('category', '')} "
                f"| {_dec_comma(y1)} | {_dec_comma(y2)} | {_dec_comma(y3)} "
                f"| {_dec_comma(y1 + y2 + y3)} |"
            )
        lines.append(
            f"| **TOTAL** | **{_dec_comma(totals[0])}** | **{_dec_comma(totals[1])}** "
            f"| **{_dec_comma(totals[2])}** | **{_dec_comma(sum(totals))}** |"
        )

        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])

        if not milestones:
            return ""

        lines = [
            "## Milestones & KPIs\n",
            "| Year | Quarter | Milestone | KPI | Target |",
            "|:----:|:-------:|-----------|-----|--------|",
        ]
        for ms in milestones:
            lines.append(
                f"| {ms.get('year', '')} "
                f"| {ms.get('quarter', '')} "
                f"| {ms.get('milestone', '')} "
                f"| {ms.get('kpi', '')} "
                f"| {ms.get('target', '')} |"
            )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*3-year decarbonization roadmap for SMEs.*"
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
            f"h3{{color:#388e3c;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:14px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.7em;color:#558b2f;text-transform:uppercase;}}"
            f".card-value{{font-size:1.3em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.65em;color:#689f38;}}"
            f".year-section{{background:{_LIGHTER};border:1px solid {_CARD_BG};"
            f"border-radius:10px;padding:16px;margin:16px 0;}}"
            f".year-badge{{display:inline-block;background:{_PRIMARY};color:#fff;"
            f"padding:4px 14px;border-radius:16px;font-weight:700;font-size:0.9em;}}"
            f".trajectory-bar{{height:28px;background:#e0e0e0;border-radius:6px;"
            f"overflow:hidden;margin:4px 0;position:relative;}}"
            f".traj-fill{{height:100%;border-radius:6px;display:flex;align-items:center;"
            f"padding-left:10px;color:#fff;font-weight:600;font-size:0.8em;"
            f"transition:width 0.5s ease;}}"
            f".macc-bar{{display:flex;align-items:center;margin:4px 0;gap:8px;}}"
            f".macc-label{{width:180px;min-width:180px;font-size:0.85em;text-align:right;"
            f"font-weight:600;color:{_PRIMARY};}}"
            f".macc-track{{flex:1;height:24px;background:#f0f0f0;border-radius:4px;"
            f"position:relative;overflow:visible;}}"
            f".macc-fill-neg{{position:absolute;right:50%;height:100%;background:#4caf50;"
            f"border-radius:4px 0 0 4px;}}"
            f".macc-fill-pos{{position:absolute;left:50%;height:100%;background:#ff9800;"
            f"border-radius:0 4px 4px 0;}}"
            f".macc-center{{position:absolute;left:50%;top:0;bottom:0;width:2px;"
            f"background:{_PRIMARY};}}"
            f".milestone-card{{background:{_LIGHT};border-left:4px solid {_ACCENT};"
            f"padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;font-size:0.9em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.summary-cards{{grid-template-columns:1fr 1fr;}}"
            f".macc-label{{width:100px;min-width:100px;font-size:0.75em;}}"
            f".report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>3-Year Decarbonization Roadmap</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'Baseline: {_dec_comma(data.get("baseline_tco2e", 0))} tCO2e | '
            f'Generated: {ts}</p>'
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        baseline = float(data.get("baseline_tco2e", 0))
        currency = data.get("currency", "GBP")
        cumulative = 0.0
        total_inv = 0.0
        total_sav = 0.0

        for yr in [1, 2, 3]:
            yr_data = data.get(f"year{yr}", {})
            actions = yr_data.get("actions", [])
            cumulative += sum(float(a.get("reduction_tco2e", 0)) for a in actions)
            total_inv += sum(float(a.get("cost", 0)) for a in actions)
            total_sav += sum(float(a.get("annual_savings", 0)) for a in actions)

        return (
            f'<h2>Roadmap Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Baseline</div>'
            f'<div class="card-value">{_dec_comma(baseline)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">3yr Reduction</div>'
            f'<div class="card-value">{_dec_comma(cumulative)}</div>'
            f'<div class="card-unit">tCO2e ({_pct(_safe_div(cumulative, baseline) * 100)})</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Investment</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_inv)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual Savings</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_sav)}</div></div>\n'
            f'</div>'
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        baseline = float(data.get("baseline_tco2e", 0))
        if baseline <= 0:
            return ""

        bars = ""
        cumulative = 0.0
        colors = [_ACCENT, "#66bb6a", "#a5d6a7", "#c8e6c9"]

        for yr in [0, 1, 2, 3]:
            if yr == 0:
                pct = 100
                label = f"Baseline: {_dec_comma(baseline)} tCO2e"
            else:
                yr_data = data.get(f"year{yr}", {})
                yr_red = sum(float(a.get("reduction_tco2e", 0))
                             for a in yr_data.get("actions", []))
                cumulative += yr_red
                remaining = baseline - cumulative
                pct = max(_safe_div(remaining, baseline) * 100, 3)
                label = f"Year {yr}: {_dec_comma(remaining)} tCO2e (-{_pct(_safe_div(cumulative, baseline) * 100)})"

            bars += (
                f'<p style="margin:2px 0;font-size:0.85em;font-weight:600;">'
                f'{"Baseline" if yr == 0 else "Year " + str(yr)}</p>\n'
                f'<div class="trajectory-bar">'
                f'<div class="traj-fill" style="width:{pct:.1f}%;'
                f'background:{colors[yr]}">{label}</div></div>\n'
            )

        return f'<h2>Emissions Trajectory</h2>\n{bars}'

    def _html_year_detail(self, data: Dict[str, Any], year: int) -> str:
        yr_data = data.get(f"year{year}", {})
        theme = yr_data.get("theme", f"Year {year}")
        actions = yr_data.get("actions", [])
        currency = data.get("currency", "GBP")

        total_red = sum(float(a.get("reduction_tco2e", 0)) for a in actions)
        total_inv = sum(float(a.get("cost", 0)) for a in actions)
        total_sav = sum(float(a.get("annual_savings", 0)) for a in actions)

        rows = ""
        for a in actions:
            red = float(a.get("reduction_tco2e", 0))
            cost = float(a.get("cost", 0))
            rows += (
                f'<tr><td>{a.get("name", "")}</td>'
                f'<td>{a.get("category", "")}</td>'
                f'<td>{_dec_comma(red)} tCO2e</td>'
                f'<td>{currency} {_dec_comma(cost)}</td>'
                f'<td>{currency} {_dec_comma(a.get("annual_savings", 0))}</td>'
                f'<td>{currency} {_dec_comma(_safe_div(cost, red))}</td></tr>\n'
            )

        return (
            f'<div class="year-section">\n'
            f'<h2><span class="year-badge">Year {year}</span> {theme}</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Actions</div>'
            f'<div class="card-value">{len(actions)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction</div>'
            f'<div class="card-value">{_dec_comma(total_red)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Investment</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_inv)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Savings/yr</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_sav)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Action</th><th>Category</th><th>Reduction</th>'
            f'<th>Cost</th><th>Savings/yr</th><th>Cost/tCO2e</th></tr>\n'
            f'{rows}</table>\n'
            f'</div>'
        )

    def _html_macc_curve(self, data: Dict[str, Any]) -> str:
        macc_data = data.get("macc_data", [])
        currency = data.get("currency", "GBP")

        if not macc_data:
            return f'<h2>MACC Curve</h2><p>No MACC data provided.</p>'

        sorted_macc = sorted(macc_data, key=lambda x: float(x.get("cost_per_tco2e", 0)))
        max_cost = max(abs(float(m.get("cost_per_tco2e", 0))) for m in sorted_macc) if sorted_macc else 1

        bars = ""
        for m in sorted_macc:
            cost = float(m.get("cost_per_tco2e", 0))
            name = m.get("name", "")[:25]

            if cost < 0:
                width_pct = min(abs(cost) / max_cost * 48, 48)
                bars += (
                    f'<div class="macc-bar">'
                    f'<div class="macc-label">{name}</div>'
                    f'<div class="macc-track">'
                    f'<div class="macc-center"></div>'
                    f'<div class="macc-fill-neg" style="width:{width_pct:.1f}%"></div>'
                    f'</div>'
                    f'<span style="font-size:0.75em;min-width:80px;">'
                    f'{currency} {_dec_comma(cost)}/tCO2e</span>'
                    f'</div>\n'
                )
            else:
                width_pct = min(cost / max_cost * 48, 48)
                bars += (
                    f'<div class="macc-bar">'
                    f'<div class="macc-label">{name}</div>'
                    f'<div class="macc-track">'
                    f'<div class="macc-center"></div>'
                    f'<div class="macc-fill-pos" style="width:{width_pct:.1f}%"></div>'
                    f'</div>'
                    f'<span style="font-size:0.75em;min-width:80px;">'
                    f'{currency} {_dec_comma(cost)}/tCO2e</span>'
                    f'</div>\n'
                )

        return (
            f'<h2>Marginal Abatement Cost Curve</h2>\n'
            f'<p style="font-size:0.85em;color:#689f38;">'
            f'Left of center = saves money | Right of center = costs money</p>\n'
            f'{bars}'
        )

    def _html_budget_allocation(self, data: Dict[str, Any]) -> str:
        categories = data.get("budget_categories", [])
        currency = data.get("currency", "GBP")

        if not categories:
            return ""

        rows = ""
        totals = [0.0, 0.0, 0.0]
        for cat in categories:
            y1 = float(cat.get("year1", 0))
            y2 = float(cat.get("year2", 0))
            y3 = float(cat.get("year3", 0))
            totals[0] += y1
            totals[1] += y2
            totals[2] += y3
            rows += (
                f'<tr><td>{cat.get("category", "")}</td>'
                f'<td>{currency} {_dec_comma(y1)}</td>'
                f'<td>{currency} {_dec_comma(y2)}</td>'
                f'<td>{currency} {_dec_comma(y3)}</td>'
                f'<td><strong>{currency} {_dec_comma(y1 + y2 + y3)}</strong></td></tr>\n'
            )
        rows += (
            f'<tr><th>TOTAL</th>'
            f'<th>{currency} {_dec_comma(totals[0])}</th>'
            f'<th>{currency} {_dec_comma(totals[1])}</th>'
            f'<th>{currency} {_dec_comma(totals[2])}</th>'
            f'<th>{currency} {_dec_comma(sum(totals))}</th></tr>\n'
        )

        return (
            f'<h2>Budget Allocation</h2>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Year 1</th><th>Year 2</th>'
            f'<th>Year 3</th><th>Total</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        if not milestones:
            return ""

        cards = ""
        for ms in milestones:
            cards += (
                f'<div class="milestone-card">'
                f'<strong>Y{ms.get("year", "")} Q{ms.get("quarter", "")}</strong> '
                f'{ms.get("milestone", "")} '
                f'<span style="color:#689f38;">| KPI: {ms.get("kpi", "")} '
                f'Target: {ms.get("target", "")}</span>'
                f'</div>\n'
            )

        return f'<h2>Milestones & KPIs</h2>\n{cards}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'3-year decarbonization roadmap for SMEs'
            f'</div>'
        )
