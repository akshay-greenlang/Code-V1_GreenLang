# -*- coding: utf-8 -*-
"""
AbatementWaterfallReportTemplate - Sector abatement waterfall for PACK-028.

Renders a sector-specific abatement waterfall report with lever-by-lever
breakdown, cost curves, implementation timelines, lever interdependencies,
and cumulative abatement analysis. Multi-format (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Sector Abatement Overview
    3.  Waterfall Chart Data
    4.  Lever-by-Lever Breakdown
    5.  Cost Curve Analysis (EUR/tCO2e)
    6.  Implementation Timeline
    7.  Lever Interdependencies
    8.  Cumulative Abatement Projection
    9.  Quick Wins Analysis
    10. Sensitivity Analysis
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "abatement_waterfall_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_WARN = "#ef6c00"
_DANGER = "#c62828"
_SUCCESS = "#2e7d32"

SECTOR_LEVERS: Dict[str, List[Dict[str, Any]]] = {
    "power": [
        {"id": "renewable_expansion", "name": "Renewable Capacity Expansion", "category": "Fuel Switch"},
        {"id": "coal_phaseout", "name": "Coal Plant Phase-Out", "category": "Fuel Switch"},
        {"id": "gas_efficiency", "name": "Gas Peaking Plant Efficiency", "category": "Efficiency"},
        {"id": "grid_storage", "name": "Grid Energy Storage", "category": "Storage"},
        {"id": "demand_response", "name": "Demand Response & Smart Grid", "category": "Efficiency"},
        {"id": "nuclear", "name": "Nuclear Capacity (SMR)", "category": "Low-Carbon"},
        {"id": "ccs_fossil", "name": "CCS for Fossil Generation", "category": "CCS"},
    ],
    "steel": [
        {"id": "bf_efficiency", "name": "Blast Furnace Efficiency", "category": "Efficiency"},
        {"id": "eaf_transition", "name": "EAF Transition", "category": "Process Change"},
        {"id": "dri_hydrogen", "name": "Green Hydrogen DRI", "category": "Hydrogen"},
        {"id": "ccs_integrated", "name": "CCS for Integrated Plants", "category": "CCS"},
        {"id": "scrap_recycling", "name": "Scrap Recycling Increase", "category": "Circular"},
        {"id": "waste_heat", "name": "Waste Heat Recovery", "category": "Efficiency"},
    ],
    "cement": [
        {"id": "clinker_sub", "name": "Clinker Substitution", "category": "Process Change"},
        {"id": "alt_fuels", "name": "Alternative Fuels (Biomass/Waste)", "category": "Fuel Switch"},
        {"id": "kiln_efficiency", "name": "High-Efficiency Kiln", "category": "Efficiency"},
        {"id": "ccus", "name": "Carbon Capture & Storage", "category": "CCS"},
        {"id": "low_carbon", "name": "Low-Carbon Cement Products", "category": "Product"},
        {"id": "circular", "name": "Concrete Reuse (Circular)", "category": "Circular"},
    ],
}

XBRL_ABATEMENT_TAGS: Dict[str, str] = {
    "total_abatement": "gl:TotalAbatementPotential",
    "total_cost": "gl:TotalAbatementCost",
    "lever_count": "gl:AbatementLeverCount",
    "avg_cost_per_tco2e": "gl:AverageMarginalAbatementCost",
    "quick_win_count": "gl:QuickWinLeverCount",
    "quick_win_abatement": "gl:QuickWinAbatementPotential",
}


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
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        neg = int_part.startswith("-")
        if neg:
            int_part = int_part[1:]
        fmt = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                fmt = "," + fmt
            fmt = ch + fmt
        if neg:
            fmt = "-" + fmt
        if len(parts) > 1:
            fmt += "." + parts[1]
        return fmt
    except Exception:
        return str(val)


def _pct_of(part: Any, total: Any) -> str:
    p = Decimal(str(part))
    t = Decimal(str(total))
    if t == 0:
        return "0.00"
    r = (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(r)


class AbatementWaterfallReportTemplate:
    """
    Sector abatement waterfall report template.

    Renders lever-by-lever abatement breakdown with waterfall chart data,
    marginal abatement cost curves, implementation timelines, and
    interdependency mapping. Supports MD, HTML, JSON, and PDF.

    Example:
        >>> template = AbatementWaterfallReportTemplate()
        >>> data = {
        ...     "org_name": "CementCo",
        ...     "sector_id": "cement",
        ...     "baseline_emissions": 1250000,
        ...     "levers": [
        ...         {"name": "Clinker Substitution", "abatement_tco2e": 175000, ...},
        ...     ],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_sector_overview(data),
            self._md_waterfall(data),
            self._md_lever_breakdown(data),
            self._md_cost_curve(data),
            self._md_timeline(data),
            self._md_interdependencies(data),
            self._md_cumulative(data),
            self._md_quick_wins(data),
            self._md_sensitivity(data),
            self._md_xbrl_tags(data),
            self._md_audit_trail(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_sector_overview(data),
            self._html_waterfall(data),
            self._html_lever_breakdown(data),
            self._html_cost_curve(data),
            self._html_timeline(data),
            self._html_interdependencies(data),
            self._html_cumulative(data),
            self._html_quick_wins(data),
            self._html_sensitivity(data),
            self._html_xbrl_tags(data),
            self._html_audit_trail(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Abatement Waterfall - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        levers = data.get("levers", [])
        baseline = float(data.get("baseline_emissions", 0))
        total_abatement = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        total_cost = sum(float(l.get("total_cost", 0)) for l in levers)
        avg_mac = (total_cost / total_abatement) if total_abatement > 0 else 0
        quick_wins = [l for l in levers if l.get("quick_win", False) or float(l.get("payback_years", 99)) <= 2]

        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector_id": data.get("sector_id", ""),
            "baseline_emissions_tco2e": str(baseline),
            "summary": {
                "total_abatement_tco2e": str(total_abatement),
                "abatement_share_pct": _pct_of(total_abatement, baseline),
                "residual_emissions": str(baseline - total_abatement),
                "total_cost_eur": str(total_cost),
                "average_mac_eur_per_tco2e": str(round(avg_mac, 2)),
                "lever_count": len(levers),
                "quick_win_count": len(quick_wins),
            },
            "waterfall": self._build_waterfall(data),
            "levers": levers,
            "quick_wins": quick_wins,
            "interdependencies": data.get("interdependencies", []),
            "sensitivity": data.get("sensitivity", {}),
            "xbrl_tags": {k: XBRL_ABATEMENT_TAGS[k] for k in XBRL_ABATEMENT_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Abatement Waterfall - {data.get('org_name', '')}",
                "author": "GreenLang PACK-028",
                "subject": "Sector Abatement Waterfall Analysis",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    def _build_waterfall(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        levers = data.get("levers", [])
        baseline = float(data.get("baseline_emissions", 0))
        waterfall = [{"label": "Baseline Emissions", "value": baseline, "type": "start", "running_total": baseline}]
        running = baseline
        sorted_levers = sorted(levers, key=lambda l: float(l.get("cost_per_tco2e", 0)))
        for l in sorted_levers:
            abatement = float(l.get("abatement_tco2e", 0))
            running -= abatement
            waterfall.append({
                "label": l.get("name", ""),
                "value": -abatement,
                "type": "reduction",
                "running_total": running,
                "cost_per_tco2e": float(l.get("cost_per_tco2e", 0)),
            })
        waterfall.append({"label": "Residual Emissions", "value": running, "type": "end", "running_total": running})
        return waterfall

    def _get_levers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sector_id = data.get("sector_id", "")
        return data.get("levers", SECTOR_LEVERS.get(sector_id, []))

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Abatement Waterfall Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {data.get('sector_id', '').replace('_', ' ').title()}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        baseline = float(data.get("baseline_emissions", 0))
        total_abatement = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        total_cost = sum(float(l.get("total_cost", 0)) for l in levers)
        avg_mac = (total_cost / total_abatement) if total_abatement > 0 else 0
        residual = baseline - total_abatement
        quick_wins = [l for l in levers if l.get("quick_win", False) or float(l.get("payback_years", 99)) <= 2]
        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |",
            f"|-----|-------|",
            f"| Baseline Emissions | {_dec_comma(baseline)} tCO2e |",
            f"| Total Abatement Potential | {_dec_comma(total_abatement)} tCO2e ({_pct_of(total_abatement, baseline)}%) |",
            f"| Residual Emissions | {_dec_comma(residual)} tCO2e |",
            f"| Number of Levers | {len(levers)} |",
            f"| Total Investment | EUR {_dec_comma(total_cost)} |",
            f"| Average MAC | EUR {_dec(avg_mac)}/tCO2e |",
            f"| Quick Wins (payback <= 2yr) | {len(quick_wins)} |",
        ]
        return "\n".join(lines)

    def _md_sector_overview(self, data: Dict[str, Any]) -> str:
        sector_id = data.get("sector_id", "")
        default_levers = SECTOR_LEVERS.get(sector_id, [])
        lines = [
            "## 2. Sector Abatement Overview\n",
            f"**Sector:** {sector_id.replace('_', ' ').title()}  \n"
            f"**Available Lever Categories:** {', '.join(sorted(set(l.get('category', '') for l in default_levers)))}\n",
            "### Sector-Specific Levers\n",
            "| # | Lever | Category |",
            "|---|-------|----------|",
        ]
        for i, l in enumerate(default_levers, 1):
            lines.append(f"| {i} | {l.get('name', '')} | {l.get('category', '')} |")
        return "\n".join(lines)

    def _md_waterfall(self, data: Dict[str, Any]) -> str:
        waterfall = self._build_waterfall(data)
        baseline = float(data.get("baseline_emissions", 0))
        lines = [
            "## 3. Waterfall Chart Data\n",
            "| Step | Label | Change (tCO2e) | Running Total | Share of Baseline |",
            "|------|-------|---------------:|--------------:|------------------:|",
        ]
        for i, w in enumerate(waterfall):
            val = w["value"]
            rt = w["running_total"]
            share = _pct_of(rt, baseline) if baseline > 0 else "0"
            if w["type"] == "start":
                lines.append(f"| {i} | **{w['label']}** | {_dec_comma(val)} | {_dec_comma(rt)} | 100.00% |")
            elif w["type"] == "end":
                lines.append(f"| {i} | **{w['label']}** | - | **{_dec_comma(rt)}** | **{share}%** |")
            else:
                lines.append(f"| {i} | {w['label']} | {_dec_comma(val)} | {_dec_comma(rt)} | {share}% |")
        return "\n".join(lines)

    def _md_lever_breakdown(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        baseline = float(data.get("baseline_emissions", 0))
        total_abatement = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        lines = [
            "## 4. Lever-by-Lever Breakdown\n",
            "| # | Lever | Abatement (tCO2e) | Share of Total | Cost (EUR/tCO2e) | Investment (EUR) | Payback (yr) |",
            "|---|-------|------------------:|---------------:|-----------------:|-----------------:|-------------:|",
        ]
        sorted_levers = sorted(levers, key=lambda l: float(l.get("abatement_tco2e", 0)), reverse=True)
        for i, l in enumerate(sorted_levers, 1):
            abatement = float(l.get("abatement_tco2e", 0))
            share = _pct_of(abatement, total_abatement) if total_abatement > 0 else "0"
            lines.append(
                f"| {i} | {l.get('name', '')} | {_dec_comma(abatement)} "
                f"| {share}% | {_dec(l.get('cost_per_tco2e', 0))} "
                f"| {_dec_comma(l.get('total_cost', 0))} "
                f"| {_dec(l.get('payback_years', 0), 1)} |"
            )
        if levers:
            lines.append(
                f"| | **Total** | **{_dec_comma(total_abatement)}** "
                f"| **100%** | **{_dec(sum(float(l.get('total_cost', 0)) for l in levers) / max(1, total_abatement))}** "
                f"| **{_dec_comma(sum(float(l.get('total_cost', 0)) for l in levers))}** | |"
            )
        return "\n".join(lines)

    def _md_cost_curve(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        sorted_by_cost = sorted(levers, key=lambda l: float(l.get("cost_per_tco2e", 0)))
        lines = [
            "## 5. Cost Curve Analysis\n",
            "Levers sorted by marginal abatement cost (EUR/tCO2e):\n",
            "| Rank | Lever | MAC (EUR/tCO2e) | Abatement (tCO2e) | Cumulative (tCO2e) | Net Cost |",
            "|------|-------|----------------:|------------------:|-------------------:|---------|",
        ]
        cumulative = 0
        for i, l in enumerate(sorted_by_cost, 1):
            mac = float(l.get("cost_per_tco2e", 0))
            abatement = float(l.get("abatement_tco2e", 0))
            cumulative += abatement
            net = "Saving" if mac < 0 else ("Neutral" if mac == 0 else "Cost")
            lines.append(
                f"| {i} | {l.get('name', '')} | {_dec(mac)} | {_dec_comma(abatement)} "
                f"| {_dec_comma(cumulative)} | {net} |"
            )
        neg_cost = sum(float(l.get("abatement_tco2e", 0)) for l in levers if float(l.get("cost_per_tco2e", 0)) < 0)
        if neg_cost > 0:
            lines.append(f"\n**Negative-cost abatement (net savings):** {_dec_comma(neg_cost)} tCO2e")
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        lines = [
            "## 6. Implementation Timeline\n",
            "| Lever | Start | End | Duration | Phase |",
            "|-------|------:|----:|---------:|-------|",
        ]
        for l in levers:
            start = l.get("start_year", "")
            end = l.get("end_year", "")
            duration = ""
            phase = l.get("phase", "")
            if start and end:
                try:
                    duration = f"{int(end) - int(start)} years"
                except (ValueError, TypeError):
                    duration = "-"
            lines.append(f"| {l.get('name', '')} | {start} | {end} | {duration} | {phase} |")
        return "\n".join(lines)

    def _md_interdependencies(self, data: Dict[str, Any]) -> str:
        deps = data.get("interdependencies", [])
        lines = [
            "## 7. Lever Interdependencies\n",
        ]
        if deps:
            lines.append("| Lever | Depends On | Relationship | Notes |")
            lines.append("|-------|-----------|-------------|-------|")
            for d in deps:
                lines.append(
                    f"| {d.get('lever', '')} | {d.get('depends_on', '')} "
                    f"| {d.get('relationship', 'prerequisite')} | {d.get('notes', '')} |"
                )
        else:
            lines.append("_No interdependencies defined. Levers assumed independent._")
        return "\n".join(lines)

    def _md_cumulative(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        baseline = float(data.get("baseline_emissions", 0))
        years = sorted(set(
            yr for l in levers
            for yr in range(int(l.get("start_year", 2025)), int(l.get("end_year", 2050)) + 1)
            if l.get("start_year") and l.get("end_year")
        ))
        if not years:
            years = list(range(2025, 2051))
        lines = [
            "## 8. Cumulative Abatement Projection\n",
            "| Year | Cumulative Abatement (tCO2e) | Remaining (tCO2e) | Pathway Progress (%) |",
            "|------|-----------------------------:|-------------------:|---------------------:|",
        ]
        total_abatement = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        for yr in years[::5] if len(years) > 10 else years:
            active_abatement = 0
            for l in levers:
                s = int(l.get("start_year", 9999))
                e = int(l.get("end_year", 9999))
                abt = float(l.get("abatement_tco2e", 0))
                if s <= yr:
                    if yr >= e:
                        active_abatement += abt
                    else:
                        duration = max(1, e - s)
                        progress = (yr - s) / duration
                        active_abatement += abt * progress
            remaining = baseline - active_abatement
            pct = _pct_of(active_abatement, total_abatement) if total_abatement > 0 else "0"
            lines.append(f"| {yr} | {_dec_comma(active_abatement)} | {_dec_comma(remaining)} | {pct}% |")
        return "\n".join(lines)

    def _md_quick_wins(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        quick = [l for l in levers if l.get("quick_win", False) or float(l.get("payback_years", 99)) <= 2]
        lines = [
            "## 9. Quick Wins Analysis\n",
            f"**Quick wins identified:** {len(quick)} levers with payback <= 2 years\n",
        ]
        if quick:
            lines.append("| Lever | Abatement (tCO2e) | Cost (EUR/tCO2e) | Payback | Net Savings |")
            lines.append("|-------|------------------:|----------------:|--------:|------------:|")
            total_qw_abt = 0
            for l in quick:
                abt = float(l.get("abatement_tco2e", 0))
                total_qw_abt += abt
                mac = float(l.get("cost_per_tco2e", 0))
                savings = -mac * abt if mac < 0 else 0
                lines.append(
                    f"| {l.get('name', '')} | {_dec_comma(abt)} | {_dec(mac)} "
                    f"| {_dec(l.get('payback_years', 0), 1)} yr | EUR {_dec_comma(savings)} |"
                )
            lines.append(f"\n**Total Quick Win Abatement:** {_dec_comma(total_qw_abt)} tCO2e")
        else:
            lines.append("_No quick wins identified (payback > 2 years for all levers)._")
        return "\n".join(lines)

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity", {})
        scenarios = sensitivity.get("scenarios", [])
        lines = [
            "## 10. Sensitivity Analysis\n",
        ]
        if scenarios:
            lines.append("| Scenario | Total Abatement | Total Cost | MAC | Key Assumption |")
            lines.append("|----------|----------------:|----------:|----:|---------------|")
            for s in scenarios:
                lines.append(
                    f"| {s.get('name', '')} | {_dec_comma(s.get('total_abatement', 0))} tCO2e "
                    f"| EUR {_dec_comma(s.get('total_cost', 0))} "
                    f"| EUR {_dec(s.get('avg_mac', 0))}/tCO2e | {s.get('assumption', '')} |"
                )
        else:
            lines.append(
                "| Base Case | As calculated | As calculated | As calculated | Standard assumptions |\n"
                "| High Carbon Price | +10-15% abatement | Higher | Lower effective MAC | Carbon price EUR 150+/tCO2e |\n"
                "| Technology Delay | -5-10% abatement | Higher | Higher MAC | 2-3 year tech delay |\n"
                "| Accelerated Policy | +15-20% abatement | Lower | Lower MAC | Aggressive policy support |"
            )
        return "\n".join(lines)

    def _md_xbrl_tags(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        total_abt = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        total_cost = sum(float(l.get("total_cost", 0)) for l in levers)
        avg_mac = (total_cost / total_abt) if total_abt > 0 else 0
        quick = [l for l in levers if l.get("quick_win", False) or float(l.get("payback_years", 99)) <= 2]
        qw_abt = sum(float(l.get("abatement_tco2e", 0)) for l in quick)
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
            f"| Total Abatement | {XBRL_ABATEMENT_TAGS['total_abatement']} | {_dec_comma(total_abt)} tCO2e |",
            f"| Total Cost | {XBRL_ABATEMENT_TAGS['total_cost']} | EUR {_dec_comma(total_cost)} |",
            f"| Lever Count | {XBRL_ABATEMENT_TAGS['lever_count']} | {len(levers)} |",
            f"| Average MAC | {XBRL_ABATEMENT_TAGS['avg_cost_per_tco2e']} | EUR {_dec(avg_mac)}/tCO2e |",
            f"| Quick Win Count | {XBRL_ABATEMENT_TAGS['quick_win_count']} | {len(quick)} |",
            f"| Quick Win Abatement | {XBRL_ABATEMENT_TAGS['quick_win_abatement']} | {_dec_comma(qw_abt)} tCO2e |",
        ]
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Input Hash | `{dh[:16]}...` |\n"
            f"| Engine | Deterministic (zero-hallucination) |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n"
            f"*Sector abatement waterfall with lever-by-lever cost curves.*"
        )

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".waterfall-bar{{display:inline-block;height:20px;border-radius:3px;}}"
            f".wf-start{{background:{_PRIMARY};}}.wf-reduction{{background:{_ACCENT};}}.wf-end{{background:{_WARN};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Abatement Waterfall Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name", "")} | <strong>Sector:</strong> {data.get("sector_id", "").replace("_", " ").title()} | <strong>Generated:</strong> {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        baseline = float(data.get("baseline_emissions", 0))
        total_abt = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        total_cost = sum(float(l.get("total_cost", 0)) for l in levers)
        avg_mac = (total_cost / total_abt) if total_abt > 0 else 0
        residual = baseline - total_abt
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Baseline</div><div class="card-value">{_dec_comma(baseline)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Total Abatement</div><div class="card-value">{_dec_comma(total_abt)}</div><div class="card-unit">tCO2e ({_pct_of(total_abt, baseline)}%)</div></div>\n'
            f'<div class="card"><div class="card-label">Residual</div><div class="card-value">{_dec_comma(residual)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Avg MAC</div><div class="card-value">EUR {_dec(avg_mac)}</div><div class="card-unit">/tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Levers</div><div class="card-value">{len(levers)}</div></div>\n'
            f'</div>'
        )

    def _html_sector_overview(self, data: Dict[str, Any]) -> str:
        sector_id = data.get("sector_id", "")
        default_levers = SECTOR_LEVERS.get(sector_id, [])
        rows = "".join(f'<tr><td>{i}</td><td>{l["name"]}</td><td>{l["category"]}</td></tr>\n' for i, l in enumerate(default_levers, 1))
        return f'<h2>2. Sector Overview</h2>\n<table>\n<tr><th>#</th><th>Lever</th><th>Category</th></tr>\n{rows}</table>'

    def _html_waterfall(self, data: Dict[str, Any]) -> str:
        wf = self._build_waterfall(data)
        baseline = float(data.get("baseline_emissions", 0))
        rows = ""
        for w in wf:
            val = w["value"]
            rt = w["running_total"]
            bar_pct = (abs(val) / baseline * 100) if baseline > 0 else 0
            cls = "wf-start" if w["type"] == "start" else ("wf-end" if w["type"] == "end" else "wf-reduction")
            rows += (
                f'<tr><td>{w["label"]}</td><td>{_dec_comma(val)}</td><td>{_dec_comma(rt)}</td>'
                f'<td><div class="waterfall-bar {cls}" style="width:{min(bar_pct, 100)}%">&nbsp;</div></td></tr>\n'
            )
        return f'<h2>3. Waterfall</h2>\n<table>\n<tr><th>Step</th><th>Change</th><th>Running Total</th><th>Visual</th></tr>\n{rows}</table>'

    def _html_lever_breakdown(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        total_abt = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        sorted_l = sorted(levers, key=lambda l: float(l.get("abatement_tco2e", 0)), reverse=True)
        rows = ""
        for i, l in enumerate(sorted_l, 1):
            abt = float(l.get("abatement_tco2e", 0))
            rows += (
                f'<tr><td>{i}</td><td>{l.get("name", "")}</td><td>{_dec_comma(abt)}</td>'
                f'<td>{_pct_of(abt, total_abt)}%</td><td>EUR {_dec(l.get("cost_per_tco2e", 0))}</td>'
                f'<td>EUR {_dec_comma(l.get("total_cost", 0))}</td></tr>\n'
            )
        return (
            f'<h2>4. Lever Breakdown</h2>\n<table>\n'
            f'<tr><th>#</th><th>Lever</th><th>Abatement</th><th>Share</th><th>MAC</th><th>Investment</th></tr>\n{rows}</table>'
        )

    def _html_cost_curve(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        sorted_l = sorted(levers, key=lambda l: float(l.get("cost_per_tco2e", 0)))
        rows = ""
        cum = 0
        for i, l in enumerate(sorted_l, 1):
            abt = float(l.get("abatement_tco2e", 0))
            cum += abt
            rows += (
                f'<tr><td>{i}</td><td>{l.get("name", "")}</td><td>EUR {_dec(l.get("cost_per_tco2e", 0))}</td>'
                f'<td>{_dec_comma(abt)}</td><td>{_dec_comma(cum)}</td></tr>\n'
            )
        return f'<h2>5. Cost Curve</h2>\n<table>\n<tr><th>#</th><th>Lever</th><th>MAC</th><th>Abatement</th><th>Cumulative</th></tr>\n{rows}</table>'

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        rows = ""
        for l in levers:
            s, e = l.get("start_year", ""), l.get("end_year", "")
            dur = f"{int(e) - int(s)} yr" if s and e else "-"
            rows += f'<tr><td>{l.get("name", "")}</td><td>{s}</td><td>{e}</td><td>{dur}</td><td>{l.get("phase", "")}</td></tr>\n'
        return f'<h2>6. Timeline</h2>\n<table>\n<tr><th>Lever</th><th>Start</th><th>End</th><th>Duration</th><th>Phase</th></tr>\n{rows}</table>'

    def _html_interdependencies(self, data: Dict[str, Any]) -> str:
        deps = data.get("interdependencies", [])
        rows = "".join(f'<tr><td>{d.get("lever","")}</td><td>{d.get("depends_on","")}</td><td>{d.get("relationship","")}</td><td>{d.get("notes","")}</td></tr>\n' for d in deps)
        return f'<h2>7. Interdependencies</h2>\n<table>\n<tr><th>Lever</th><th>Depends On</th><th>Relationship</th><th>Notes</th></tr>\n{rows}</table>'

    def _html_cumulative(self, data: Dict[str, Any]) -> str:
        return f'<h2>8. Cumulative Projection</h2>\n<p><em>See Markdown or JSON output for detailed year-by-year cumulative projections.</em></p>'

    def _html_quick_wins(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        quick = [l for l in levers if l.get("quick_win", False) or float(l.get("payback_years", 99)) <= 2]
        rows = ""
        for l in quick:
            abt = float(l.get("abatement_tco2e", 0))
            mac = float(l.get("cost_per_tco2e", 0))
            savings = -mac * abt if mac < 0 else 0
            rows += (
                f'<tr><td>{l.get("name","")}</td><td>{_dec_comma(abt)}</td>'
                f'<td>EUR {_dec(mac)}</td><td>{_dec(l.get("payback_years",0),1)} yr</td>'
                f'<td>EUR {_dec_comma(savings)}</td></tr>\n'
            )
        return (
            f'<h2>9. Quick Wins</h2>\n<p>{len(quick)} levers with payback &lt;= 2 years</p>\n'
            f'<table>\n<tr><th>Lever</th><th>Abatement</th><th>MAC</th><th>Payback</th><th>Savings</th></tr>\n{rows}</table>'
        )

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        return f'<h2>10. Sensitivity</h2>\n<p><em>See Markdown output for scenario sensitivity analysis.</em></p>'

    def _html_xbrl_tags(self, data: Dict[str, Any]) -> str:
        levers = self._get_levers(data)
        total_abt = sum(float(l.get("abatement_tco2e", 0)) for l in levers)
        total_cost = sum(float(l.get("total_cost", 0)) for l in levers)
        avg_mac = (total_cost / total_abt) if total_abt > 0 else 0
        return (
            f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n'
            f'<tr><td>Total Abatement</td><td><code>{XBRL_ABATEMENT_TAGS["total_abatement"]}</code></td><td>{_dec_comma(total_abt)} tCO2e</td></tr>\n'
            f'<tr><td>Total Cost</td><td><code>{XBRL_ABATEMENT_TAGS["total_cost"]}</code></td><td>EUR {_dec_comma(total_cost)}</td></tr>\n'
            f'<tr><td>Avg MAC</td><td><code>{XBRL_ABATEMENT_TAGS["avg_cost_per_tco2e"]}</code></td><td>EUR {_dec(avg_mac)}/tCO2e</td></tr>\n'
            f'</table>'
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n'
            f'<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-028 on {ts} - Sector abatement waterfall analysis</div>'
