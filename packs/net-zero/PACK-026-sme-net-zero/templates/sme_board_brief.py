# -*- coding: utf-8 -*-
"""
SMEBoardBriefTemplate - 1-page executive summary for board/leadership (PACK-026).

Renders a concise board-level briefing covering current emissions state,
targets, quick wins, grant opportunities, investment requirements,
risk/opportunity summary, and a recommended decision with approve/reject.
Optimised for PDF-ready layout in 1 page.

Sections:
    1. Current State (baseline, intensity, peer comparison)
    2. Targets (2030 + 2050 commitments)
    3. Quick Wins (top 3 with ROI)
    4. Grant Opportunities (funding available)
    5. Investment Required (cost + returns)
    6. Risk & Opportunity Summary
    7. Recommended Decision (approve Y/N)

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
_TEMPLATE_ID = "sme_board_brief"

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


# ===========================================================================
# Template Class
# ===========================================================================

class SMEBoardBriefTemplate:
    """
    SME board briefing template - 1-page executive summary.

    Renders a concise board-level document covering current state,
    targets, quick wins, grants, investment, risks/opportunities,
    and a recommended decision across Markdown, HTML, and JSON formats.
    Optimised for PDF-ready, single-page layout.
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
        """Render the board brief as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_current_state(data),
            self._md_targets(data),
            self._md_quick_wins(data),
            self._md_grants(data),
            self._md_investment(data),
            self._md_risks_opportunities(data),
            self._md_recommendation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the board brief as HTML (PDF-optimised layout)."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_current_state(data),
            self._html_targets(data),
            self._html_quick_wins(data),
            self._html_grants(data),
            self._html_investment(data),
            self._html_risks_opportunities(data),
            self._html_recommendation(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>Board Brief - Net Zero</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the board brief as structured JSON."""
        self.generated_at = _utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))

        quick_wins = data.get("quick_wins", [])[:3]
        grants = data.get("grants", [])
        risks = data.get("risks", [])
        opportunities = data.get("opportunities", [])

        total_investment = float(data.get("total_investment", 0))
        total_annual_savings = float(data.get("total_annual_savings", 0))
        grant_funding_min = sum(float(g.get("funding_min", 0)) for g in grants)
        grant_funding_max = sum(float(g.get("funding_max", 0)) for g in grants)

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
            },
            "current_state": {
                "total_emissions_tco2e": round(total, 2),
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "intensity_per_employee": round(_safe_div(total, employees), 2),
                "sector_average_tco2e": data.get("sector_avg_tco2e", 0),
                "percentile_rank": data.get("percentile_rank", 50),
            },
            "targets": {
                "target_2030_pct": data.get("target_2030_pct", 42),
                "target_2030_tco2e": round(total * (1 - float(data.get("target_2030_pct", 42)) / 100), 2),
                "target_2050": data.get("target_2050", "Net Zero"),
                "framework": data.get("target_framework", "SBTi SME"),
            },
            "top_quick_wins": [
                {
                    "name": qw.get("name", ""),
                    "reduction_tco2e": qw.get("reduction_tco2e", 0),
                    "cost": qw.get("cost", 0),
                    "annual_savings": qw.get("annual_savings", 0),
                    "payback_months": qw.get("payback_months", 0),
                }
                for qw in quick_wins
            ],
            "grant_funding": {
                "grants_matched": len(grants),
                "total_min": round(grant_funding_min, 2),
                "total_max": round(grant_funding_max, 2),
                "currency": data.get("currency", "GBP"),
            },
            "investment": {
                "total_required": round(total_investment, 2),
                "annual_savings": round(total_annual_savings, 2),
                "five_year_net": round(total_annual_savings * 5 - total_investment, 2),
                "currency": data.get("currency", "GBP"),
            },
            "risks": risks,
            "opportunities": opportunities,
            "recommendation": {
                "decision": data.get("recommendation", "APPROVE"),
                "rationale": data.get("recommendation_rationale", ""),
            },
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
            f"# Board Briefing: Net Zero Strategy\n\n"
            f"**Organization:** {org}  \n"
            f"**Prepared For:** Board of Directors / Senior Leadership  \n"
            f"**Date:** {ts}  \n"
            f"**Classification:** Confidential\n\n---"
        )

    def _md_current_state(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)
        avg = float(data.get("sector_avg_tco2e", 0))
        pctile = int(data.get("percentile_rank", 50))
        position = "Below average (good)" if total < avg else "Above average"

        return (
            f"## 1. Current State\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Total Emissions | **{_dec_comma(total)} tCO2e** |\n"
            f"| Scope 1 / 2 / 3 | {_dec_comma(s1)} / {_dec_comma(s2)} / {_dec_comma(s3)} tCO2e |\n"
            f"| Intensity | {_dec(intensity)} tCO2e per employee |\n"
            f"| Sector Average | {_dec_comma(avg)} tCO2e |\n"
            f"| Percentile | {pctile}th ({position}) |"
        )

    def _md_targets(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        t2030 = float(data.get("target_2030_pct", 42))
        target_2030_tco2e = total * (1 - t2030 / 100)
        framework = data.get("target_framework", "SBTi SME Target")

        return (
            f"## 2. Targets\n\n"
            f"| Target | Commitment |\n"
            f"|--------|------------|\n"
            f"| **2030 Target** | Reduce by {_pct(t2030)} to {_dec_comma(target_2030_tco2e)} tCO2e |\n"
            f"| **2050 Target** | {data.get('target_2050', 'Net Zero')} |\n"
            f"| Framework | {framework} |\n"
            f"| Aligned to | 1.5C pathway |"
        )

    def _md_quick_wins(self, data: Dict[str, Any]) -> str:
        wins = data.get("quick_wins", [])[:3]
        currency = data.get("currency", "GBP")

        lines = [
            "## 3. Quick Wins (Top 3)\n",
            f"| Action | Reduction | Cost | Savings/yr | Payback |",
            f"|--------|----------:|-----:|----------:|--------:|",
        ]
        for w in wins:
            lines.append(
                f"| {w.get('name', '')} "
                f"| {_dec_comma(w.get('reduction_tco2e', 0))} tCO2e "
                f"| {currency} {_dec_comma(w.get('cost', 0))} "
                f"| {currency} {_dec_comma(w.get('annual_savings', 0))} "
                f"| {w.get('payback_months', 0)} months |"
            )
        return "\n".join(lines)

    def _md_grants(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")
        total_min = sum(float(g.get("funding_min", 0)) for g in grants)
        total_max = sum(float(g.get("funding_max", 0)) for g in grants)

        lines = [
            "## 4. Grant Opportunities\n",
            f"**{len(grants)} grants matched** | Potential: "
            f"{currency} {_dec_comma(total_min)} - {currency} {_dec_comma(total_max)}\n",
        ]
        for g in grants[:3]:
            score = float(g.get("eligibility_score", 0))
            lines.append(
                f"- **{g.get('name', '')}** ({g.get('funding_body', '')}): "
                f"{currency} {_dec_comma(g.get('funding_min', 0))} - "
                f"{_dec_comma(g.get('funding_max', 0))} | "
                f"Eligibility: {_dec(score, 0)}/100 | "
                f"Deadline: {g.get('application_deadline', '')}"
            )

        return "\n".join(lines)

    def _md_investment(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        total_inv = float(data.get("total_investment", 0))
        annual_sav = float(data.get("total_annual_savings", 0))
        five_yr_net = annual_sav * 5 - total_inv
        grants_max = sum(float(g.get("funding_max", 0)) for g in data.get("grants", []))
        net_cost = total_inv - grants_max

        return (
            f"## 5. Investment Required\n\n"
            f"| Item | Amount |\n"
            f"|------|-------:|\n"
            f"| Total Investment | {currency} {_dec_comma(total_inv)} |\n"
            f"| Less: Grant Funding (max) | -{currency} {_dec_comma(grants_max)} |\n"
            f"| **Net Cost to Company** | **{currency} {_dec_comma(max(net_cost, 0))}** |\n"
            f"| Annual Savings | {currency} {_dec_comma(annual_sav)} |\n"
            f"| **5-Year Net Benefit** | **{currency} {_dec_comma(five_yr_net)}** |"
        )

    def _md_risks_opportunities(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [
            {"category": "Climate", "description": "Increasing regulatory requirements", "impact": "Medium"},
            {"category": "Reputation", "description": "Customer expectations for sustainability", "impact": "High"},
            {"category": "Cost", "description": "Rising energy prices", "impact": "High"},
        ])
        opportunities = data.get("opportunities", [
            {"category": "Cost Savings", "description": "Energy efficiency reduces operating costs", "impact": "High"},
            {"category": "Revenue", "description": "Green products command premium pricing", "impact": "Medium"},
            {"category": "Talent", "description": "Attract employees who value sustainability", "impact": "Medium"},
        ])

        lines = [
            "## 6. Risks & Opportunities\n",
            "**Risks:**\n",
            "| Category | Description | Impact |",
            "|----------|-------------|:------:|",
        ]
        for r in risks:
            lines.append(f"| {r.get('category', '')} | {r.get('description', '')} | {r.get('impact', '')} |")

        lines.append("\n**Opportunities:**\n")
        lines.append("| Category | Description | Impact |")
        lines.append("|----------|-------------|:------:|")
        for o in opportunities:
            lines.append(f"| {o.get('category', '')} | {o.get('description', '')} | {o.get('impact', '')} |")

        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        decision = data.get("recommendation", "APPROVE")
        rationale = data.get("recommendation_rationale",
                             "The financial case is positive with a net benefit over 5 years. "
                             "Grant funding further reduces the investment required. "
                             "Action now positions the company ahead of regulatory requirements "
                             "and customer expectations.")

        return (
            f"## 7. Recommended Decision\n\n"
            f"**Decision:** **{decision}**\n\n"
            f"**Rationale:** {rationale}\n\n"
            f"---\n\n"
            f"| Approved By | Signature | Date |\n"
            f"|-------------|-----------|------|\n"
            f"| | | |\n"
            f"| | | |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Board briefing document - confidential.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"@page{{size:A4;margin:15mm;}}"
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.5;font-size:0.9em;}}"
            f".report{{max-width:800px;margin:0 auto;background:#fff;padding:28px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:10px;"
            f"font-size:1.5em;margin-bottom:8px;}}"
            f"h2{{color:{_SECONDARY};margin-top:18px;border-left:4px solid {_ACCENT};"
            f"padding-left:10px;font-size:1.05em;}}"
            f"table{{width:100%;border-collapse:collapse;margin:8px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:6px 10px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));"
            f"gap:8px;margin:10px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:8px;"
            f"padding:10px;text-align:center;border-left:3px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.65em;color:#558b2f;text-transform:uppercase;}}"
            f".card-value{{font-size:1.3em;font-weight:700;color:{_PRIMARY};margin-top:2px;}}"
            f".card-unit{{font-size:0.65em;color:#689f38;}}"
            f".decision-box{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});"
            f"border:3px solid {_PRIMARY};border-radius:12px;padding:16px;text-align:center;"
            f"margin:16px 0;}}"
            f".decision-label{{font-size:0.85em;color:#558b2f;text-transform:uppercase;}}"
            f".decision-value{{font-size:1.8em;font-weight:800;color:{_PRIMARY};}}"
            f".risk-opp{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;}}"
            f".risk-col,.opp-col{{padding:10px;border-radius:8px;}}"
            f".risk-col{{background:#fff3e0;border-left:4px solid #ff9800;}}"
            f".opp-col{{background:{_LIGHTER};border-left:4px solid {_ACCENT};}}"
            f".risk-col h3{{color:#e65100;font-size:0.9em;margin:0 0 6px;}}"
            f".opp-col h3{{color:{_PRIMARY};font-size:0.9em;margin:0 0 6px;}}"
            f".item{{font-size:0.8em;padding:3px 0;border-bottom:1px solid rgba(0,0,0,0.05);}}"
            f".impact{{display:inline-block;padding:1px 6px;border-radius:4px;font-size:0.7em;"
            f"font-weight:600;color:#fff;margin-left:4px;}}"
            f".impact-high{{background:#f44336;}}"
            f".impact-medium{{background:#ff9800;}}"
            f".impact-low{{background:#9e9e9e;}}"
            f".sig-table{{margin-top:12px;}}"
            f".sig-table td{{height:30px;}}"
            f".footer{{margin-top:20px;padding-top:10px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.75em;text-align:center;}}"
            f"@media print{{body{{background:#fff;padding:0;}}"
            f".report{{box-shadow:none;border-radius:0;padding:0;}}}}"
            f"@media(max-width:600px){{.risk-opp{{grid-template-columns:1fr;}}"
            f".summary-cards{{grid-template-columns:1fr 1fr;}}.report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Your Company")
        return (
            f'<h1>Board Briefing: Net Zero Strategy</h1>\n'
            f'<p><strong>{org}</strong> | '
            f'Board of Directors / Senior Leadership | '
            f'{ts} | Confidential</p>'
        )

    def _html_current_state(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        intensity = _safe_div(total, employees)
        pctile = int(data.get("percentile_rank", 50))

        return (
            f'<h2>1. Current State</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Emissions</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Per Employee</div>'
            f'<div class="card-value">{_dec(intensity)}</div>'
            f'<div class="card-unit">tCO2e/FTE</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 1 / 2 / 3</div>'
            f'<div class="card-value" style="font-size:0.9em;">'
            f'{_dec_comma(s1)} / {_dec_comma(s2)} / {_dec_comma(s3)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Percentile</div>'
            f'<div class="card-value">{pctile}th</div>'
            f'<div class="card-unit">in sector</div></div>\n'
            f'</div>'
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        t2030 = float(data.get("target_2030_pct", 42))
        target_2030_abs = total * (1 - t2030 / 100)

        return (
            f'<h2>2. Targets</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">2030 Target</div>'
            f'<div class="card-value">-{_pct(t2030)}</div>'
            f'<div class="card-unit">{_dec_comma(target_2030_abs)} tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">2050 Target</div>'
            f'<div class="card-value">{data.get("target_2050", "Net Zero")}</div>'
            f'<div class="card-unit">{data.get("target_framework", "SBTi SME")}</div></div>\n'
            f'</div>'
        )

    def _html_quick_wins(self, data: Dict[str, Any]) -> str:
        wins = data.get("quick_wins", [])[:3]
        currency = data.get("currency", "GBP")

        rows = ""
        for w in wins:
            rows += (
                f'<tr><td><strong>{w.get("name", "")}</strong></td>'
                f'<td>{_dec_comma(w.get("reduction_tco2e", 0))} tCO2e</td>'
                f'<td>{currency} {_dec_comma(w.get("cost", 0))}</td>'
                f'<td>{currency} {_dec_comma(w.get("annual_savings", 0))}</td>'
                f'<td>{w.get("payback_months", 0)} mo</td></tr>\n'
            )

        return (
            f'<h2>3. Quick Wins (Top 3)</h2>\n'
            f'<table>\n'
            f'<tr><th>Action</th><th>Reduction</th><th>Cost</th>'
            f'<th>Savings/yr</th><th>Payback</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_grants(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")
        total_max = sum(float(g.get("funding_max", 0)) for g in grants)

        rows = ""
        for g in grants[:3]:
            rows += (
                f'<tr><td>{g.get("name", "")}</td>'
                f'<td>{g.get("funding_body", "")}</td>'
                f'<td>{currency} {_dec_comma(g.get("funding_min", 0))} - '
                f'{_dec_comma(g.get("funding_max", 0))}</td>'
                f'<td>{g.get("application_deadline", "")}</td></tr>\n'
            )

        return (
            f'<h2>4. Grant Opportunities ({len(grants)} matched, '
            f'up to {currency} {_dec_comma(total_max)})</h2>\n'
            f'<table>\n'
            f'<tr><th>Grant</th><th>Funder</th><th>Amount</th><th>Deadline</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_investment(self, data: Dict[str, Any]) -> str:
        currency = data.get("currency", "GBP")
        total_inv = float(data.get("total_investment", 0))
        annual_sav = float(data.get("total_annual_savings", 0))
        grants_max = sum(float(g.get("funding_max", 0)) for g in data.get("grants", []))
        net_cost = max(total_inv - grants_max, 0)
        five_yr_net = annual_sav * 5 - total_inv

        return (
            f'<h2>5. Investment Required</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Investment</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_inv)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Less Grants</div>'
            f'<div class="card-value">-{currency} {_dec_comma(grants_max)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net Cost</div>'
            f'<div class="card-value">{currency} {_dec_comma(net_cost)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual Savings</div>'
            f'<div class="card-value">{currency} {_dec_comma(annual_sav)}</div></div>\n'
            f'  <div class="card"><div class="card-label">5yr Net Benefit</div>'
            f'<div class="card-value">{currency} {_dec_comma(five_yr_net)}</div></div>\n'
            f'</div>'
        )

    def _html_risks_opportunities(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [
            {"category": "Climate", "description": "Increasing regulatory requirements", "impact": "Medium"},
            {"category": "Reputation", "description": "Customer expectations", "impact": "High"},
            {"category": "Cost", "description": "Rising energy prices", "impact": "High"},
        ])
        opportunities = data.get("opportunities", [
            {"category": "Cost Savings", "description": "Energy efficiency", "impact": "High"},
            {"category": "Revenue", "description": "Green products premium", "impact": "Medium"},
            {"category": "Talent", "description": "Attract green talent", "impact": "Medium"},
        ])

        risk_items = ""
        for r in risks:
            impact = r.get("impact", "Medium").lower()
            risk_items += (
                f'<div class="item">{r.get("category", "")}: {r.get("description", "")}'
                f' <span class="impact impact-{impact}">{r.get("impact", "")}</span></div>\n'
            )

        opp_items = ""
        for o in opportunities:
            impact = o.get("impact", "Medium").lower()
            opp_items += (
                f'<div class="item">{o.get("category", "")}: {o.get("description", "")}'
                f' <span class="impact impact-{impact}">{o.get("impact", "")}</span></div>\n'
            )

        return (
            f'<h2>6. Risks & Opportunities</h2>\n'
            f'<div class="risk-opp">\n'
            f'  <div class="risk-col"><h3>Risks</h3>{risk_items}</div>\n'
            f'  <div class="opp-col"><h3>Opportunities</h3>{opp_items}</div>\n'
            f'</div>'
        )

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        decision = data.get("recommendation", "APPROVE")
        rationale = data.get("recommendation_rationale",
                             "The financial case is positive with a net benefit over 5 years. "
                             "Grant funding further reduces the investment. "
                             "Action now positions the company ahead of regulatory requirements.")

        return (
            f'<h2>7. Recommended Decision</h2>\n'
            f'<div class="decision-box">\n'
            f'  <div class="decision-label">Recommendation</div>\n'
            f'  <div class="decision-value">{decision}</div>\n'
            f'  <p style="font-size:0.85em;color:#558b2f;">{rationale}</p>\n'
            f'</div>\n'
            f'<table class="sig-table">\n'
            f'<tr><th>Approved By</th><th>Signature</th><th>Date</th></tr>\n'
            f'<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>\n'
            f'<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts} | Confidential'
            f'</div>'
        )
