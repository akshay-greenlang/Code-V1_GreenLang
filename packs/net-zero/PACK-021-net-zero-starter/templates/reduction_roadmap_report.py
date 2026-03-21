# -*- coding: utf-8 -*-
"""
ReductionRoadmapReportTemplate - Phased reduction roadmap with MACC for PACK-021.

Renders a phased decarbonization roadmap with emissions hotspot analysis,
abatement options, Marginal Abatement Cost Curve data, phased implementation,
investment summary, cumulative impact projections, quick wins, and
implementation dependencies.

Sections:
    1. Emissions Hotspot Analysis
    2. Abatement Options Summary
    3. MACC Curve Data (sorted by cost)
    4. Phased Roadmap (short/medium/long)
    5. Investment Summary (CapEx/OpEx by phase)
    6. Cumulative Impact Projection
    7. Quick Wins (payback < 2yr)
    8. Implementation Dependencies

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


class ReductionRoadmapReportTemplate:
    """
    Phased reduction roadmap with MACC data.

    Renders a comprehensive abatement roadmap covering hotspot analysis,
    action prioritization via Marginal Abatement Cost Curve, phased
    implementation, investment analysis, quick wins, and dependencies.

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
            self._md_hotspots(data),
            self._md_abatement_options(data),
            self._md_macc(data),
            self._md_phased_roadmap(data),
            self._md_investment(data),
            self._md_cumulative_impact(data),
            self._md_quick_wins(data),
            self._md_dependencies(data),
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
            self._html_hotspots(data),
            self._html_abatement_options(data),
            self._html_macc(data),
            self._html_phased_roadmap(data),
            self._html_investment(data),
            self._html_cumulative_impact(data),
            self._html_quick_wins(data),
            self._html_dependencies(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Reduction Roadmap Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        actions = data.get("abatement_options", [])
        macc_sorted = sorted(actions, key=lambda a: Decimal(str(a.get("cost_per_tco2e", 0))))
        total_abatement = sum(Decimal(str(a.get("abatement_tco2e", 0))) for a in actions)
        total_capex = sum(Decimal(str(a.get("capex_eur", 0))) for a in actions)

        result: Dict[str, Any] = {
            "template": "reduction_roadmap_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "hotspots": data.get("hotspots", []),
            "abatement_options": actions,
            "macc_curve": [
                {
                    "name": a.get("name", ""),
                    "cost_per_tco2e": str(Decimal(str(a.get("cost_per_tco2e", 0)))),
                    "abatement_tco2e": str(Decimal(str(a.get("abatement_tco2e", 0)))),
                }
                for a in macc_sorted
            ],
            "phased_roadmap": data.get("phases", []),
            "investment_summary": {
                "total_capex_eur": str(total_capex),
                "phases": data.get("investment_phases", []),
            },
            "cumulative_impact": data.get("cumulative_impact", []),
            "quick_wins": [
                a for a in actions
                if Decimal(str(a.get("payback_years", 99))) < Decimal("2")
            ],
            "dependencies": data.get("dependencies", []),
            "summary": {
                "total_abatement_tco2e": str(total_abatement),
                "total_capex_eur": str(total_capex),
                "actions_count": len(actions),
            },
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Reduction Roadmap Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_hotspots(self, data: Dict[str, Any]) -> str:
        hotspots = data.get("hotspots", [])
        total_emissions = sum(Decimal(str(h.get("emissions_tco2e", 0))) for h in hotspots)
        lines = [
            "## 1. Emissions Hotspot Analysis\n",
            "| # | Source | Scope | Emissions (tCO2e) | Share (%) | Priority |",
            "|---|--------|-------|------------------:|----------:|----------|",
        ]
        for i, h in enumerate(hotspots, 1):
            emissions = Decimal(str(h.get("emissions_tco2e", 0)))
            lines.append(
                f"| {i} | {h.get('source', '-')} | {h.get('scope', '-')} "
                f"| {_dec_comma(emissions)} "
                f"| {_dec(_pct_of(emissions, total_emissions))}% "
                f"| {h.get('priority', '-')} |"
            )
        if not hotspots:
            lines.append("| - | _No hotspots identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_abatement_options(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        lines = [
            "## 2. Abatement Options Summary\n",
            "| # | Action | Scope | Abatement (tCO2e/yr) | Cost (EUR/tCO2e) | TRL | Timeline | Payback (yr) |",
            "|---|--------|-------|---------------------:|----------------:|----:|----------|-------------:|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('name', '-')} | {a.get('scope', '-')} "
                f"| {_dec_comma(a.get('abatement_tco2e', 0))} "
                f"| {_dec_comma(a.get('cost_per_tco2e', 0))} "
                f"| {a.get('trl', '-')} | {a.get('timeline', '-')} "
                f"| {_dec(a.get('payback_years', 0))} |"
            )
        if not actions:
            lines.append("| - | _No actions defined_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_macc(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        macc_sorted = sorted(actions, key=lambda a: Decimal(str(a.get("cost_per_tco2e", 0))))
        cumulative = Decimal("0")
        lines = [
            "## 3. MACC Curve Data\n",
            "*Sorted by marginal abatement cost (ascending).*\n",
            "| # | Action | Cost (EUR/tCO2e) | Abatement (tCO2e/yr) | Cumulative Abatement |",
            "|---|--------|----------------:|---------------------:|--------------------:|",
        ]
        for i, a in enumerate(macc_sorted, 1):
            abatement = Decimal(str(a.get("abatement_tco2e", 0)))
            cumulative += abatement
            cost = Decimal(str(a.get("cost_per_tco2e", 0)))
            lines.append(
                f"| {i} | {a.get('name', '-')} "
                f"| {_dec_comma(cost)} "
                f"| {_dec_comma(abatement)} "
                f"| {_dec_comma(cumulative)} |"
            )
        return "\n".join(lines)

    def _md_phased_roadmap(self, data: Dict[str, Any]) -> str:
        phases = data.get("phases", [])
        lines = [
            "## 4. Phased Roadmap\n",
        ]
        for phase in phases:
            phase_name = phase.get("name", "Phase")
            period = phase.get("period", "")
            lines.append(f"### {phase_name} ({period})\n")
            phase_actions = phase.get("actions", [])
            if phase_actions:
                lines.append("| Action | Abatement (tCO2e/yr) | Investment (EUR) | Status |")
                lines.append("|--------|---------------------:|----------------:|--------|")
                for a in phase_actions:
                    lines.append(
                        f"| {a.get('name', '-')} "
                        f"| {_dec_comma(a.get('abatement_tco2e', 0))} "
                        f"| {_dec_comma(a.get('investment_eur', 0), 0)} "
                        f"| {a.get('status', '-')} |"
                    )
            else:
                lines.append("_No actions assigned to this phase._")
            lines.append("")
        if not phases:
            lines.append("_No phases defined._")
        return "\n".join(lines)

    def _md_investment(self, data: Dict[str, Any]) -> str:
        inv_phases = data.get("investment_phases", [])
        total_capex = sum(Decimal(str(p.get("capex_eur", 0))) for p in inv_phases)
        total_opex = sum(Decimal(str(p.get("opex_annual_eur", 0))) for p in inv_phases)
        total_savings = sum(Decimal(str(p.get("savings_annual_eur", 0))) for p in inv_phases)
        lines = [
            "## 5. Investment Summary\n",
            f"**Total CapEx:** EUR {_dec_comma(total_capex, 0)}  \n"
            f"**Total Annual OpEx:** EUR {_dec_comma(total_opex, 0)}  \n"
            f"**Total Annual Savings:** EUR {_dec_comma(total_savings, 0)}\n",
            "| Phase | CapEx (EUR) | OpEx (EUR/yr) | Savings (EUR/yr) | Net Cost (EUR/yr) |",
            "|-------|------------:|--------------:|----------------:|-----------------:|",
        ]
        for p in inv_phases:
            capex = Decimal(str(p.get("capex_eur", 0)))
            opex = Decimal(str(p.get("opex_annual_eur", 0)))
            savings = Decimal(str(p.get("savings_annual_eur", 0)))
            net = opex - savings
            lines.append(
                f"| {p.get('name', '-')} | {_dec_comma(capex, 0)} "
                f"| {_dec_comma(opex, 0)} | {_dec_comma(savings, 0)} "
                f"| {_dec_comma(net, 0)} |"
            )
        return "\n".join(lines)

    def _md_cumulative_impact(self, data: Dict[str, Any]) -> str:
        impact = data.get("cumulative_impact", [])
        lines = [
            "## 6. Cumulative Impact Projection\n",
            "| Year | Annual Abatement (tCO2e) | Cumulative Abatement (tCO2e) | Residual (tCO2e) | % Reduced |",
            "|------|-------------------------:|-----------------------------:|-----------------:|----------:|",
        ]
        for yr in impact:
            lines.append(
                f"| {yr.get('year', '-')} "
                f"| {_dec_comma(yr.get('annual_abatement_tco2e', 0))} "
                f"| {_dec_comma(yr.get('cumulative_abatement_tco2e', 0))} "
                f"| {_dec_comma(yr.get('residual_tco2e', 0))} "
                f"| {_dec(yr.get('reduction_pct', 0))}% |"
            )
        if not impact:
            lines.append("| - | - | - | - | - |")
        return "\n".join(lines)

    def _md_quick_wins(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        quick = [
            a for a in actions
            if Decimal(str(a.get("payback_years", 99))) < Decimal("2")
        ]
        lines = [
            "## 7. Quick Wins (Payback < 2 Years)\n",
        ]
        if quick:
            lines.append("| Action | Abatement (tCO2e/yr) | Cost (EUR/tCO2e) | Payback (yr) | CapEx (EUR) |")
            lines.append("|--------|---------------------:|----------------:|-------------:|------------:|")
            for a in quick:
                lines.append(
                    f"| {a.get('name', '-')} "
                    f"| {_dec_comma(a.get('abatement_tco2e', 0))} "
                    f"| {_dec_comma(a.get('cost_per_tco2e', 0))} "
                    f"| {_dec(a.get('payback_years', 0))} "
                    f"| {_dec_comma(a.get('capex_eur', 0), 0)} |"
                )
            total_qw_abatement = sum(Decimal(str(a.get("abatement_tco2e", 0))) for a in quick)
            lines.append(f"\n**Total Quick-Win Abatement:** {_dec_comma(total_qw_abatement)} tCO2e/yr")
        else:
            lines.append("_No actions with payback under 2 years identified._")
        return "\n".join(lines)

    def _md_dependencies(self, data: Dict[str, Any]) -> str:
        deps = data.get("dependencies", [])
        lines = [
            "## 8. Implementation Dependencies\n",
            "| Action | Depends On | Type | Risk if Delayed |",
            "|--------|-----------|------|----------------|",
        ]
        for d in deps:
            lines.append(
                f"| {d.get('action', '-')} | {d.get('depends_on', '-')} "
                f"| {d.get('type', '-')} | {d.get('risk', '-')} |"
            )
        if not deps:
            lines.append("| _No dependencies identified_ | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*Marginal Abatement Cost Curve methodology applied.*"
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
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".action-card{border:1px solid #c8e6c9;border-radius:8px;padding:16px;margin:12px 0;}"
            ".action-card.priority-high{border-left:4px solid #c62828;}"
            ".action-card.priority-medium{border-left:4px solid #ff8f00;}"
            ".action-card.priority-low{border-left:4px solid #2e7d32;}"
            ".macc-negative{background:#c8e6c9;}"
            ".macc-positive{background:#fff9c4;}"
            ".quick-win{background:#e8f5e9;border-left:4px solid #43a047;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:16px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:#43a047;}"
            ".fill-amber{background:#ff8f00;}"
            ".fill-red{background:#e53935;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        actions = data.get("abatement_options", [])
        total_abatement = sum(Decimal(str(a.get("abatement_tco2e", 0))) for a in actions)
        total_capex = sum(Decimal(str(a.get("capex_eur", 0))) for a in actions)
        return (
            f'<h1>Reduction Roadmap Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Generated:</strong> {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Actions</div>'
            f'<div class="card-value">{len(actions)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Abatement</div>'
            f'<div class="card-value">{_dec_comma(total_abatement)}</div>'
            f'<div class="card-unit">tCO2e/yr</div></div>\n'
            f'  <div class="card"><div class="card-label">Total CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(total_capex, 0)}</div></div>\n'
            f'</div>'
        )

    def _html_hotspots(self, data: Dict[str, Any]) -> str:
        hotspots = data.get("hotspots", [])
        total_e = sum(Decimal(str(h.get("emissions_tco2e", 0))) for h in hotspots)
        rows = ""
        for i, h in enumerate(hotspots, 1):
            emissions = Decimal(str(h.get("emissions_tco2e", 0)))
            pct = float(_pct_of(emissions, total_e)) if total_e > 0 else 0
            rows += (
                f'<tr><td>{i}</td><td>{h.get("source", "-")}</td>'
                f'<td>{h.get("scope", "-")}</td>'
                f'<td>{_dec_comma(emissions)}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill fill-green" '
                f'style="width:{max(pct, 2)}%"></div></div> {_dec(pct)}%</td>'
                f'<td>{h.get("priority", "-")}</td></tr>\n'
            )
        return (
            f'<h2>1. Emissions Hotspot Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Source</th><th>Scope</th><th>Emissions (tCO2e)</th>'
            f'<th>Share</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_abatement_options(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("name", "-")}</td>'
                f'<td>{a.get("scope", "-")}</td>'
                f'<td>{_dec_comma(a.get("abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(a.get("cost_per_tco2e", 0))}</td>'
                f'<td>{a.get("trl", "-")}</td>'
                f'<td>{a.get("timeline", "-")}</td>'
                f'<td>{_dec(a.get("payback_years", 0))}</td></tr>\n'
            )
        return (
            f'<h2>2. Abatement Options Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Scope</th><th>Abatement (tCO2e/yr)</th>'
            f'<th>Cost (EUR/tCO2e)</th><th>TRL</th><th>Timeline</th><th>Payback (yr)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_macc(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        macc_sorted = sorted(actions, key=lambda a: Decimal(str(a.get("cost_per_tco2e", 0))))
        cumulative = Decimal("0")
        rows = ""
        for i, a in enumerate(macc_sorted, 1):
            abatement = Decimal(str(a.get("abatement_tco2e", 0)))
            cost = Decimal(str(a.get("cost_per_tco2e", 0)))
            cumulative += abatement
            cls = "macc-negative" if cost < 0 else "macc-positive"
            rows += (
                f'<tr class="{cls}"><td>{i}</td><td>{a.get("name", "-")}</td>'
                f'<td>{_dec_comma(cost)}</td>'
                f'<td>{_dec_comma(abatement)}</td>'
                f'<td>{_dec_comma(cumulative)}</td></tr>\n'
            )
        return (
            f'<h2>3. MACC Curve Data</h2>\n'
            f'<p><em>Sorted by marginal abatement cost (ascending). Negative costs represent net savings.</em></p>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Cost (EUR/tCO2e)</th>'
            f'<th>Abatement (tCO2e/yr)</th><th>Cumulative</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_phased_roadmap(self, data: Dict[str, Any]) -> str:
        phases = data.get("phases", [])
        content = '<h2>4. Phased Roadmap</h2>\n'
        for phase in phases:
            phase_name = phase.get("name", "Phase")
            period = phase.get("period", "")
            content += f'<h3>{phase_name} ({period})</h3>\n'
            phase_actions = phase.get("actions", [])
            if phase_actions:
                rows = ""
                for a in phase_actions:
                    status = a.get("status", "planned")
                    status_cls = (
                        "fill-green" if status.lower() in ("completed", "on_track")
                        else "fill-amber" if status.lower() in ("in_progress",)
                        else "fill-red"
                    )
                    rows += (
                        f'<tr><td>{a.get("name", "-")}</td>'
                        f'<td>{_dec_comma(a.get("abatement_tco2e", 0))}</td>'
                        f'<td>{_dec_comma(a.get("investment_eur", 0), 0)}</td>'
                        f'<td>{status}</td></tr>\n'
                    )
                content += (
                    f'<table>\n'
                    f'<tr><th>Action</th><th>Abatement (tCO2e/yr)</th>'
                    f'<th>Investment (EUR)</th><th>Status</th></tr>\n'
                    f'{rows}</table>\n'
                )
            else:
                content += '<p><em>No actions assigned to this phase.</em></p>\n'
        return content

    def _html_investment(self, data: Dict[str, Any]) -> str:
        inv_phases = data.get("investment_phases", [])
        total_capex = sum(Decimal(str(p.get("capex_eur", 0))) for p in inv_phases)
        total_opex = sum(Decimal(str(p.get("opex_annual_eur", 0))) for p in inv_phases)
        total_savings = sum(Decimal(str(p.get("savings_annual_eur", 0))) for p in inv_phases)
        rows = ""
        for p in inv_phases:
            capex = Decimal(str(p.get("capex_eur", 0)))
            opex = Decimal(str(p.get("opex_annual_eur", 0)))
            savings = Decimal(str(p.get("savings_annual_eur", 0)))
            net = opex - savings
            rows += (
                f'<tr><td>{p.get("name", "-")}</td>'
                f'<td>{_dec_comma(capex, 0)}</td>'
                f'<td>{_dec_comma(opex, 0)}</td>'
                f'<td>{_dec_comma(savings, 0)}</td>'
                f'<td>{_dec_comma(net, 0)}</td></tr>\n'
            )
        return (
            f'<h2>5. Investment Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(total_capex, 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual OpEx</div>'
            f'<div class="card-value">EUR {_dec_comma(total_opex, 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual Savings</div>'
            f'<div class="card-value">EUR {_dec_comma(total_savings, 0)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Phase</th><th>CapEx (EUR)</th><th>OpEx (EUR/yr)</th>'
            f'<th>Savings (EUR/yr)</th><th>Net Cost (EUR/yr)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cumulative_impact(self, data: Dict[str, Any]) -> str:
        impact = data.get("cumulative_impact", [])
        rows = ""
        for yr in impact:
            pct = float(Decimal(str(yr.get("reduction_pct", 0))))
            bar_color = "fill-green" if pct >= 50 else "fill-amber" if pct >= 25 else "fill-red"
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(yr.get("annual_abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(yr.get("cumulative_abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(yr.get("residual_tco2e", 0))}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill {bar_color}" '
                f'style="width:{min(pct, 100)}%"></div></div> {_dec(pct)}%</td></tr>\n'
            )
        return (
            f'<h2>6. Cumulative Impact Projection</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Annual Abatement</th><th>Cumulative</th>'
            f'<th>Residual</th><th>% Reduced</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_quick_wins(self, data: Dict[str, Any]) -> str:
        actions = data.get("abatement_options", [])
        quick = [
            a for a in actions
            if Decimal(str(a.get("payback_years", 99))) < Decimal("2")
        ]
        rows = ""
        for a in quick:
            rows += (
                f'<tr class="quick-win"><td>{a.get("name", "-")}</td>'
                f'<td>{_dec_comma(a.get("abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(a.get("cost_per_tco2e", 0))}</td>'
                f'<td>{_dec(a.get("payback_years", 0))}</td>'
                f'<td>{_dec_comma(a.get("capex_eur", 0), 0)}</td></tr>\n'
            )
        total_qw = sum(Decimal(str(a.get("abatement_tco2e", 0))) for a in quick)
        return (
            f'<h2>7. Quick Wins (Payback &lt; 2 Years)</h2>\n'
            f'<p><strong>{len(quick)} actions</strong> with total abatement of '
            f'<strong>{_dec_comma(total_qw)} tCO2e/yr</strong></p>\n'
            f'<table>\n'
            f'<tr><th>Action</th><th>Abatement (tCO2e/yr)</th><th>Cost (EUR/tCO2e)</th>'
            f'<th>Payback (yr)</th><th>CapEx (EUR)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_dependencies(self, data: Dict[str, Any]) -> str:
        deps = data.get("dependencies", [])
        rows = ""
        for d in deps:
            rows += (
                f'<tr><td>{d.get("action", "-")}</td>'
                f'<td>{d.get("depends_on", "-")}</td>'
                f'<td>{d.get("type", "-")}</td>'
                f'<td>{d.get("risk", "-")}</td></tr>\n'
            )
        return (
            f'<h2>8. Implementation Dependencies</h2>\n'
            f'<table>\n'
            f'<tr><th>Action</th><th>Depends On</th><th>Type</th>'
            f'<th>Risk if Delayed</th></tr>\n'
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
