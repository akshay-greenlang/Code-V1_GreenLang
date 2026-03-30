# -*- coding: utf-8 -*-
"""
AnnualReportTemplate - Annual carbon neutrality report for PACK-024.

Renders the comprehensive annual report covering the full carbon neutrality
lifecycle: footprint, reductions, credits, neutralization, claims, and
verification for the reporting year.

Sections:
    1. Executive Summary
    2. Annual Cycle Phases
    3. Emissions Summary
    4. Reduction Performance
    5. Credit Portfolio
    6. Neutralization Balance
    7. Claims & Verification
    8. Year-over-Year Trends
    9. Next Year Priorities

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "24.0.0"

def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(d):
    r = json.dumps(d, sort_keys=True, default=str) if isinstance(d, dict) else str(d)
    return hashlib.sha256(r.encode("utf-8")).hexdigest()
def _dec(v, p=2):
    try: return str(Decimal(str(v)).quantize(Decimal("0."+"0"*p), rounding=ROUND_HALF_UP))
    except: return str(v)
def _dec_comma(v, p=0):
    try:
        d = Decimal(str(v)); q = "0."+"0"*p if p > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP); parts = str(r).split(".")
        ip = parts[0]; neg = ip.startswith("-")
        if neg: ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: f = "," + f
            f = ch + f
        if neg: f = "-" + f
        if len(parts) > 1: f += "." + parts[1]
        return f
    except: return str(v)
def _pct(v):
    try: return _dec(v, 1) + "%"
    except: return str(v)

class AnnualReportTemplate:
    """Annual carbon neutrality report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive(data), self._md_phases(data),
            self._md_emissions(data), self._md_reductions(data), self._md_credits(data),
            self._md_neutralization(data), self._md_claims_verification(data),
            self._md_trends(data), self._md_priorities(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = ("body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f0f4f0;}"
               ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;}"
               "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
               "h2{color:#2e7d32;border-left:4px solid #43a047;padding-left:12px;margin-top:35px;}"
               "table{width:100%;border-collapse:collapse;margin:15px 0;}"
               "th,td{border:1px solid #c8e6c9;padding:10px;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}"
               ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
               ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
               ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Annual Carbon Neutrality Report</h1>\n{self._html_executive(data)}\n{self._html_emissions(data)}\n{self._html_neutralization(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "annual_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "metrics": data.get("metrics", {}), "is_carbon_neutral": data.get("is_carbon_neutral", False)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Annual Carbon Neutrality Report\n\n**Organization:** {org}  \n**Year:** {year}  \n**Generated:** {ts}\n\n---"

    def _md_executive(self, data):
        m = data.get("metrics", {})
        neutral = "ACHIEVED" if m.get("is_carbon_neutral", False) else "NOT ACHIEVED"
        return (f"## 1. Executive Summary\n\n"
                f"**Carbon Neutrality Status: {neutral}**\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Emissions | {_dec_comma(m.get('total_tco2e', 0))} tCO2e |\n"
                f"| Reductions Achieved | {_dec_comma(m.get('reductions_tco2e', 0))} tCO2e |\n"
                f"| Credits Retired | {_dec_comma(m.get('credits_tco2e', 0))} tCO2e |\n"
                f"| Coverage | {_pct(m.get('coverage_pct', 0))} |\n"
                f"| Credit Spend | ${_dec_comma(m.get('credit_spend_usd', 0))} |\n"
                f"| YoY Reduction | {_pct(m.get('yoy_reduction_pct', 0))} |\n"
                f"| Verification | {m.get('verification_opinion', 'N/A')} |")

    def _md_phases(self, data):
        phases = data.get("phases", [])
        lines = ["## 2. Annual Cycle Phases\n",
                  "| # | Phase | Status | Duration | Records |",
                  "|---|-------|:------:|:--------:|:-------:|"]
        for i, p in enumerate(phases, 1):
            lines.append(
                f"| {i} | {p.get('name', '-')} | {p.get('status', '-')} "
                f"| {p.get('duration', '-')} | {p.get('records', 0)} |")
        return "\n".join(lines)

    def _md_emissions(self, data):
        e = data.get("emissions", {})
        return (f"## 3. Emissions Summary\n\n| Scope | tCO2e | % |\n|-------|------:|:-:|\n"
                f"| Scope 1 | {_dec_comma(e.get('scope1', 0))} | {_pct(e.get('scope1_pct', 0))} |\n"
                f"| Scope 2 | {_dec_comma(e.get('scope2', 0))} | {_pct(e.get('scope2_pct', 0))} |\n"
                f"| Scope 3 | {_dec_comma(e.get('scope3', 0))} | {_pct(e.get('scope3_pct', 0))} |\n"
                f"| **Total** | **{_dec_comma(e.get('total', 0))}** | **100%** |")

    def _md_reductions(self, data):
        r = data.get("reductions", {})
        return (f"## 4. Reduction Performance\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Target | {_dec_comma(r.get('target_tco2e', 0))} tCO2e |\n"
                f"| Achieved | {_dec_comma(r.get('achieved_tco2e', 0))} tCO2e |\n"
                f"| Achievement Rate | {_pct(r.get('achievement_pct', 0))} |\n"
                f"| Strategies Active | {r.get('strategies_count', 0)} |")

    def _md_credits(self, data):
        c = data.get("credits", {})
        return (f"## 5. Credit Portfolio\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Volume | {_dec_comma(c.get('total_tco2e', 0))} tCO2e |\n"
                f"| Total Cost | ${_dec_comma(c.get('total_cost_usd', 0))} |\n"
                f"| Avg Price | ${_dec(c.get('avg_price', 0))} /tCO2e |\n"
                f"| Removal Share | {_pct(c.get('removal_pct', 0))} |\n"
                f"| Quality Score | {_dec(c.get('quality_score', 0), 1)}/100 |")

    def _md_neutralization(self, data):
        n = data.get("neutralization", {})
        return (f"## 6. Neutralization Balance\n\n| Item | tCO2e |\n|------|------:|\n"
                f"| Residual Emissions | {_dec_comma(n.get('residual', 0))} |\n"
                f"| Credits Retired | {_dec_comma(n.get('credits', 0))} |\n"
                f"| Net Balance | {_dec_comma(n.get('net_balance', 0))} |\n"
                f"| Coverage | {_pct(n.get('coverage_pct', 0))} |\n"
                f"| Status | {'NEUTRAL' if n.get('is_neutral', False) else 'DEFICIT'} |")

    def _md_claims_verification(self, data):
        cv = data.get("claims_verification", {})
        return (f"## 7. Claims & Verification\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Claim Type | {cv.get('claim_type', 'Carbon Neutral Organization')} |\n"
                f"| Claim Valid | {cv.get('claim_valid', 'N/A')} |\n"
                f"| Verification Body | {cv.get('verifier', 'N/A')} |\n"
                f"| Assurance Level | {cv.get('assurance_level', 'Limited')} |\n"
                f"| Opinion | {cv.get('opinion', 'N/A')} |\n"
                f"| Certificate | {cv.get('certificate', 'N/A')} |")

    def _md_trends(self, data):
        trends = data.get("trends", [])
        lines = ["## 8. Year-over-Year Trends\n",
                  "| Year | Emissions (tCO2e) | Reductions | Credits | Neutral |",
                  "|:----:|------------------:|:----------:|:-------:|:-------:|"]
        for t in trends:
            lines.append(
                f"| {t.get('year', '-')} | {_dec_comma(t.get('emissions', 0))} "
                f"| {_dec_comma(t.get('reductions', 0))} | {_dec_comma(t.get('credits', 0))} "
                f"| {'Yes' if t.get('neutral', False) else 'No'} |")
        if not trends:
            lines.append("| - | _First year -- no trend data_ | - | - | - |")
        return "\n".join(lines)

    def _md_priorities(self, data):
        priorities = data.get("priorities", [])
        lines = ["## 9. Next Year Priorities\n"]
        for i, p in enumerate(priorities, 1):
            lines.append(f"{i}. {p}")
        if not priorities:
            lines.append("_Priorities to be defined during renewal preparation._")
        return "\n".join(lines)

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_executive(self, data):
        m = data.get("metrics", {})
        return (f'<h2>Executive Summary</h2><div class="cards">'
                f'<div class="card"><div class="card-label">Total Emissions</div><div class="card-value">{_dec_comma(m.get("total_tco2e", 0))}</div>tCO2e</div>'
                f'<div class="card"><div class="card-label">Reductions</div><div class="card-value">{_dec_comma(m.get("reductions_tco2e", 0))}</div>tCO2e</div>'
                f'<div class="card"><div class="card-label">Coverage</div><div class="card-value">{_pct(m.get("coverage_pct", 0))}</div></div></div>')

    def _html_emissions(self, data):
        e = data.get("emissions", {})
        return (f'<h2>Emissions</h2><table><tr><th>Scope</th><th>tCO2e</th></tr>'
                f'<tr><td>Scope 1</td><td>{_dec_comma(e.get("scope1", 0))}</td></tr>'
                f'<tr><td>Scope 2</td><td>{_dec_comma(e.get("scope2", 0))}</td></tr>'
                f'<tr><td>Scope 3</td><td>{_dec_comma(e.get("scope3", 0))}</td></tr></table>')

    def _html_neutralization(self, data):
        n = data.get("neutralization", {})
        return (f'<h2>Neutralization</h2><div class="cards">'
                f'<div class="card"><div class="card-label">Coverage</div><div class="card-value">{_pct(n.get("coverage_pct", 0))}</div></div>'
                f'<div class="card"><div class="card-label">Status</div><div class="card-value">{"NEUTRAL" if n.get("is_neutral") else "DEFICIT"}</div></div></div>')
