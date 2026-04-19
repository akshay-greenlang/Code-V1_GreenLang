# -*- coding: utf-8 -*-
"""
NeutralizationStatementReportTemplate - Neutralization statement for PACK-024.

Renders the PAS 2060 qualifying explanatory statement with neutralization
balance sheet, emissions-to-credits matching, evidence inventory,
gap analysis, and declaration text.

Sections:
    1. Declaration Statement
    2. Neutralization Balance Sheet
    3. Emissions Coverage by Scope
    4. Credit-to-Emissions Matching
    5. Evidence Inventory
    6. Gap Analysis (if applicable)

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

class NeutralizationStatementReportTemplate:
    """Neutralization statement report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_declaration(data), self._md_balance(data),
            self._md_coverage(data), self._md_matching(data), self._md_evidence(data),
            self._md_gap(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = ("body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f0f4f0;}"
               ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;}"
               "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
               "h2{color:#2e7d32;border-left:4px solid #43a047;padding-left:12px;}"
               "table{width:100%;border-collapse:collapse;margin:15px 0;}"
               "th,td{border:1px solid #c8e6c9;padding:10px;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".declaration{background:#e8f5e9;padding:20px;border-radius:8px;border-left:4px solid #2e7d32;margin:20px 0;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Neutralization Statement</h1>\n{self._html_declaration(data)}\n{self._html_balance(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "neutralization_statement_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "balance": data.get("balance", {}), "coverage": data.get("coverage", []),
                  "is_neutral": data.get("is_neutral", False)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        period = data.get("reporting_period", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# Carbon Neutralization Statement\n\n**Organization:** {org}  \n"
                f"**Period:** {period}  \n**Generated:** {ts}  \n"
                f"**Standard:** PAS 2060:2014\n\n---")

    def _md_declaration(self, data):
        org = data.get("org_name", "Organization")
        period = data.get("reporting_period", "the reporting period")
        total = data.get("total_emissions_tco2e", 0)
        is_neutral = data.get("is_neutral", False)
        status = "ACHIEVED" if is_neutral else "NOT YET ACHIEVED"
        return (f"## 1. Declaration Statement\n\n"
                f"> **Carbon Neutrality Status: {status}**\n>\n"
                f"> {org} declares that, for {period}, total greenhouse gas emissions "
                f"of {_dec_comma(total)} tCO2e have been quantified in accordance with "
                f"the GHG Protocol Corporate Standard, reduced through documented "
                f"management actions, and {'fully offset through the retirement of ' if is_neutral else 'partially offset through '}"
                f"verified carbon credits on recognized registries, "
                f"{'achieving' if is_neutral else 'working towards'} carbon neutrality "
                f"in accordance with PAS 2060:2014.")

    def _md_balance(self, data):
        b = data.get("balance", {})
        return (f"## 2. Neutralization Balance Sheet\n\n"
                f"| Item | tCO2e |\n|------|------:|\n"
                f"| **A. Total Emissions** | **{_dec_comma(b.get('total_emissions', 0))}** |\n"
                f"|   Scope 1 | {_dec_comma(b.get('scope1', 0))} |\n"
                f"|   Scope 2 | {_dec_comma(b.get('scope2', 0))} |\n"
                f"|   Scope 3 | {_dec_comma(b.get('scope3', 0))} |\n"
                f"| **B. Reductions Achieved** | **{_dec_comma(b.get('reductions', 0))}** |\n"
                f"| **C. Residual Emissions (A-B)** | **{_dec_comma(b.get('residual', 0))}** |\n"
                f"| **D. Credits Retired** | **{_dec_comma(b.get('credits_retired', 0))}** |\n"
                f"| **E. Biogenic Removals** | **{_dec_comma(b.get('biogenic_removals', 0))}** |\n"
                f"| **F. Net Balance (C-D-E)** | **{_dec_comma(b.get('net_balance', 0))}** |\n"
                f"| **Coverage** | **{_pct(b.get('coverage_pct', 0))}** |\n"
                f"| **Status** | **{'NEUTRAL' if b.get('is_neutral', False) else 'DEFICIT'}** |")

    def _md_coverage(self, data):
        coverage = data.get("coverage", [])
        lines = ["## 3. Emissions Coverage by Scope\n",
                  "| Scope | Emissions | Reductions | Credits | Coverage | Status |",
                  "|-------|----------:|:----------:|:-------:|:--------:|:------:|"]
        for c in coverage:
            lines.append(
                f"| {c.get('scope', '-')} | {_dec_comma(c.get('emissions', 0))} "
                f"| {_dec_comma(c.get('reductions', 0))} | {_dec_comma(c.get('credits', 0))} "
                f"| {_pct(c.get('coverage_pct', 0))} | {c.get('status', '-')} |")
        return "\n".join(lines)

    def _md_matching(self, data):
        matches = data.get("credit_matching", [])
        lines = ["## 4. Credit-to-Emissions Matching\n",
                  "| Credit Source | Registry | Volume | Matched To | Vintage |",
                  "|--------------|----------|-------:|:----------:|:-------:|"]
        for m in matches:
            lines.append(
                f"| {m.get('source', '-')} | {m.get('registry', '-')} "
                f"| {_dec_comma(m.get('volume', 0))} | {m.get('matched_to', '-')} "
                f"| {m.get('vintage', '-')} |")
        if not matches:
            lines.append("| - | _See credit portfolio report for details_ | - | - | - |")
        return "\n".join(lines)

    def _md_evidence(self, data):
        evidence = data.get("evidence", [])
        lines = ["## 5. Evidence Inventory\n",
                  "| # | Evidence Item | Type | Required | Available | Reference |",
                  "|---|--------------|------|:--------:|:---------:|-----------|"]
        for i, e in enumerate(evidence, 1):
            lines.append(
                f"| {i} | {e.get('description', '-')} | {e.get('type', '-')} "
                f"| {'Yes' if e.get('required', True) else 'No'} "
                f"| {'Yes' if e.get('available', False) else 'No'} "
                f"| {e.get('reference', '-')} |")
        return "\n".join(lines)

    def _md_gap(self, data):
        gap = data.get("gap_analysis", {})
        if gap.get("total_gap", 0) <= 0:
            return "## 6. Gap Analysis\n\n_No gap identified -- neutralization balance achieved._"
        return (f"## 6. Gap Analysis\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Gap | {_dec_comma(gap.get('total_gap', 0))} tCO2e |\n"
                f"| Estimated Cost to Close | ${_dec_comma(gap.get('cost_to_close', 0))} |\n"
                f"| Recommended Action | {gap.get('recommendation', 'Procure additional credits')} |")

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_declaration(self, data):
        is_neutral = data.get("is_neutral", False)
        status = "ACHIEVED" if is_neutral else "NOT YET ACHIEVED"
        return f'<div class="declaration"><strong>Carbon Neutrality: {status}</strong></div>'

    def _html_balance(self, data):
        b = data.get("balance", {})
        return (f'<h2>Balance Sheet</h2><table>'
                f'<tr><th>Item</th><th>tCO2e</th></tr>'
                f'<tr><td>Total Emissions</td><td>{_dec_comma(b.get("total_emissions", 0))}</td></tr>'
                f'<tr><td>Credits Retired</td><td>{_dec_comma(b.get("credits_retired", 0))}</td></tr>'
                f'<tr><td>Net Balance</td><td>{_dec_comma(b.get("net_balance", 0))}</td></tr></table>')
