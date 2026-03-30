# -*- coding: utf-8 -*-
"""
CreditPortfolioReportTemplate - Carbon credit portfolio report for PACK-024.

Renders credit portfolio analysis with registry distribution, credit type
mix, quality tier breakdown, vintage analysis, SDG co-benefits mapping,
ICVCM CCP compliance, and portfolio diversification metrics.

Sections:
    1. Portfolio Overview
    2. Credit Type Distribution
    3. Registry Distribution
    4. Quality Assessment (ICVCM CCP)
    5. Vintage Analysis
    6. SDG Co-Benefits
    7. Portfolio Risk Assessment
    8. Cost Analysis

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
def _dec_comma(v, p=2):
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

class CreditPortfolioReportTemplate:
    """Carbon credit portfolio report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_overview(data), self._md_type_dist(data),
            self._md_registry_dist(data), self._md_quality(data), self._md_vintage(data),
            self._md_sdg(data), self._md_risk(data), self._md_cost(data), self._md_footer(data),
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
               "th,td{border:1px solid #c8e6c9;padding:10px;text-align:left;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Carbon Credit Portfolio Report</h1>\n{self._html_overview(data)}\n{self._html_quality(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "credit_portfolio_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "portfolio": data.get("portfolio", {}), "credits": data.get("credits", []),
                  "quality_summary": data.get("quality_summary", {})}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Carbon Credit Portfolio Report\n\n**Organization:** {org}  \n**Generated:** {ts}\n\n---"

    def _md_overview(self, data):
        p = data.get("portfolio", {})
        return (f"## 1. Portfolio Overview\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Volume | {_dec_comma(p.get('total_tco2e', 0), 0)} tCO2e |\n"
                f"| Total Cost | ${_dec_comma(p.get('total_cost_usd', 0), 0)} |\n"
                f"| Avg Price | ${_dec(p.get('avg_price', 0))} /tCO2e |\n"
                f"| Credits Count | {p.get('credits_count', 0)} |\n"
                f"| Registries | {p.get('registries_count', 0)} |\n"
                f"| Removal Share | {_pct(p.get('removal_pct', 0))} |")

    def _md_type_dist(self, data):
        types = data.get("type_distribution", [])
        lines = ["## 2. Credit Type Distribution\n",
                  "| Type | Volume (tCO2e) | % of Portfolio | Avg Price |",
                  "|------|---------------:|:--------------:|:---------:|"]
        for t in types:
            lines.append(f"| {t.get('type', '-')} | {_dec_comma(t.get('volume', 0), 0)} "
                        f"| {_pct(t.get('pct', 0))} | ${_dec(t.get('avg_price', 0))} |")
        return "\n".join(lines)

    def _md_registry_dist(self, data):
        regs = data.get("registry_distribution", [])
        lines = ["## 3. Registry Distribution\n",
                  "| Registry | Volume (tCO2e) | % | Reliability Score |",
                  "|----------|---------------:|:-:|:-----------------:|"]
        for r in regs:
            lines.append(f"| {r.get('registry', '-')} | {_dec_comma(r.get('volume', 0), 0)} "
                        f"| {_pct(r.get('pct', 0))} | {_dec(r.get('reliability', 0), 0)}/100 |")
        return "\n".join(lines)

    def _md_quality(self, data):
        qs = data.get("quality_summary", {})
        return (f"## 4. Quality Assessment (ICVCM CCP)\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Portfolio Quality Score | {_dec(qs.get('overall_score', 0), 1)}/100 |\n"
                f"| Premium Credits | {_pct(qs.get('premium_pct', 0))} |\n"
                f"| Standard Credits | {_pct(qs.get('standard_pct', 0))} |\n"
                f"| CCP-Aligned | {_pct(qs.get('ccp_aligned_pct', 0))} |")

    def _md_vintage(self, data):
        vintages = data.get("vintage_analysis", [])
        lines = ["## 5. Vintage Analysis\n",
                  "| Vintage Year | Volume (tCO2e) | % of Portfolio |",
                  "|:------------:|---------------:|:--------------:|"]
        for v in vintages:
            lines.append(f"| {v.get('year', '-')} | {_dec_comma(v.get('volume', 0), 0)} | {_pct(v.get('pct', 0))} |")
        return "\n".join(lines)

    def _md_sdg(self, data):
        sdgs = data.get("sdg_contributions", [])
        lines = ["## 6. SDG Co-Benefits\n",
                  "| SDG | Name | Credits Contributing |",
                  "|:---:|------|:--------------------:|"]
        for s in sdgs:
            lines.append(f"| SDG {s.get('number', '-')} | {s.get('name', '-')} | {s.get('credits_count', 0)} |")
        if not sdgs:
            lines.append("| - | _No SDG data available_ | - |")
        return "\n".join(lines)

    def _md_risk(self, data):
        risk = data.get("risk_assessment", {})
        return (f"## 7. Portfolio Risk Assessment\n\n| Risk Factor | Level | Mitigation |\n|-------------|:-----:|------------|\n"
                f"| Permanence Risk | {risk.get('permanence', 'moderate')} | Buffer pool allocation |\n"
                f"| Concentration Risk | {risk.get('concentration', 'low')} | Multi-registry diversification |\n"
                f"| Price Volatility | {risk.get('price_volatility', 'moderate')} | Forward contracts |\n"
                f"| Regulatory Risk | {risk.get('regulatory', 'low')} | CCP-aligned credits |")

    def _md_cost(self, data):
        cost = data.get("cost_analysis", {})
        return (f"## 8. Cost Analysis\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Spend | ${_dec_comma(cost.get('total_usd', 0), 0)} |\n"
                f"| Avg Price (Avoidance) | ${_dec(cost.get('avg_avoidance', 0))} /tCO2e |\n"
                f"| Avg Price (Removal) | ${_dec(cost.get('avg_removal', 0))} /tCO2e |\n"
                f"| Budget Utilization | {_pct(cost.get('budget_util_pct', 0))} |")

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_overview(self, data):
        p = data.get("portfolio", {})
        return (f'<h2>Portfolio Overview</h2><table><tr><th>Metric</th><th>Value</th></tr>'
                f'<tr><td>Total Volume</td><td>{_dec_comma(p.get("total_tco2e", 0), 0)} tCO2e</td></tr>'
                f'<tr><td>Total Cost</td><td>${_dec_comma(p.get("total_cost_usd", 0), 0)}</td></tr></table>')

    def _html_quality(self, data):
        qs = data.get("quality_summary", {})
        return (f'<h2>Quality Assessment</h2><table><tr><th>Metric</th><th>Value</th></tr>'
                f'<tr><td>Quality Score</td><td>{_dec(qs.get("overall_score", 0), 1)}/100</td></tr></table>')
