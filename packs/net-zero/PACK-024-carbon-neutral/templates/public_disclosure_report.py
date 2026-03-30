# -*- coding: utf-8 -*-
"""
PublicDisclosureReportTemplate - Public disclosure report for PACK-024.

Renders the public-facing carbon neutrality disclosure document required
by PAS 2060, EU Green Claims Directive, and VCMI Claims Code.  Designed
for external stakeholder communication with appropriate detail level.

Sections:
    1. Carbon Neutrality Declaration
    2. Our Commitment
    3. Emissions Profile
    4. Reduction Actions
    5. Offsetting Strategy
    6. Verification
    7. Methodology & Standards
    8. Contact & Further Information

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

class PublicDisclosureReportTemplate:
    """Public disclosure report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_declaration(data), self._md_commitment(data),
            self._md_emissions(data), self._md_reductions(data), self._md_offsetting(data),
            self._md_verification(data), self._md_methodology(data), self._md_contact(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = ("body{font-family:'Segoe UI',sans-serif;padding:0;margin:0;background:#f0f4f0;}"
               ".report{max-width:900px;margin:40px auto;background:#fff;padding:50px;border-radius:16px;"
               "box-shadow:0 4px 20px rgba(0,0,0,0.1);}"
               "h1{color:#1b5e20;font-size:2.2em;text-align:center;border-bottom:3px solid #2e7d32;padding-bottom:16px;}"
               "h2{color:#2e7d32;margin-top:40px;font-size:1.4em;}"
               ".declaration{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);padding:30px;border-radius:12px;"
               "text-align:center;margin:30px 0;border-left:5px solid #2e7d32;}"
               ".declaration-title{font-size:1.6em;font-weight:700;color:#1b5e20;}"
               ".declaration-status{font-size:1.2em;color:#2e7d32;margin-top:8px;}"
               "table{width:100%;border-collapse:collapse;margin:15px 0;}"
               "th,td{border:1px solid #c8e6c9;padding:12px;text-align:left;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0;}"
               ".card{background:#f1f8e9;border-radius:10px;padding:20px;text-align:center;}"
               ".card-value{font-size:1.8em;font-weight:700;color:#1b5e20;}"
               ".card-label{font-size:0.85em;color:#558b2f;margin-top:4px;}"
               ".footer{margin-top:50px;padding-top:20px;border-top:2px solid #c8e6c9;"
               "color:#689f38;font-size:0.85em;text-align:center;}"
               ".sdg-badges span{display:inline-block;background:#e8f5e9;border:1px solid #a5d6a7;"
               "border-radius:16px;padding:4px 12px;margin:2px;font-size:0.85em;color:#2e7d32;}")
        body = "\n".join([
            self._html_header(data), self._html_declaration(data),
            self._html_emissions(data), self._html_offsetting(data),
            self._html_verification(data), self._html_footer(data),
        ])
        return f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<title>Carbon Neutrality Disclosure</title>\n<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n</body>\n</html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "public_disclosure_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "org_name": data.get("org_name", ""), "reporting_year": data.get("reporting_year", ""),
                  "is_carbon_neutral": data.get("is_carbon_neutral", False),
                  "total_emissions_tco2e": data.get("total_emissions_tco2e", 0),
                  "credits_retired_tco2e": data.get("credits_retired_tco2e", 0)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        return f"# Carbon Neutrality Disclosure\n## {org} | {year}\n\n---"

    def _md_declaration(self, data):
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        total = data.get("total_emissions_tco2e", 0)
        is_neutral = data.get("is_carbon_neutral", False)
        status = "has achieved" if is_neutral else "is working towards"
        return (f"## Carbon Neutrality Declaration\n\n"
                f"> {org} {status} carbon neutrality for the calendar year {year}. "
                f"Total greenhouse gas emissions of {_dec_comma(total)} tCO2e have been "
                f"quantified in accordance with the GHG Protocol Corporate Standard, "
                f"reduced through documented management actions, and "
                f"{'fully' if is_neutral else 'partially'} offset through the retirement "
                f"of verified carbon credits on recognized registries, in accordance "
                f"with PAS 2060:2014.\n\n"
                f"This declaration is supported by independent verification.")

    def _md_commitment(self, data):
        commitments = data.get("commitments", [
            "Reduce absolute Scope 1 and 2 emissions year-over-year",
            "Improve data quality and expand Scope 3 coverage annually",
            "Invest in high-quality carbon credits with verified co-benefits",
            "Increase the share of carbon removal credits in our portfolio",
            "Submit to independent third-party verification annually",
        ])
        lines = ["## Our Commitment\n"]
        for c in commitments:
            lines.append(f"- {c}")
        return "\n".join(lines)

    def _md_emissions(self, data):
        e = data.get("emissions", {})
        return (f"## Our Emissions Profile\n\n"
                f"| Scope | Description | Emissions (tCO2e) | Share |\n"
                f"|-------|-------------|------------------:|:-----:|\n"
                f"| Scope 1 | Direct emissions | {_dec_comma(e.get('scope1', 0))} | {_pct(e.get('scope1_pct', 0))} |\n"
                f"| Scope 2 | Purchased energy | {_dec_comma(e.get('scope2', 0))} | {_pct(e.get('scope2_pct', 0))} |\n"
                f"| Scope 3 | Value chain | {_dec_comma(e.get('scope3', 0))} | {_pct(e.get('scope3_pct', 0))} |\n"
                f"| **Total** | | **{_dec_comma(e.get('total', 0))}** | **100%** |")

    def _md_reductions(self, data):
        r = data.get("reduction_highlights", [])
        lines = ["## What We Are Doing to Reduce Emissions\n"]
        if r:
            for item in r:
                lines.append(f"- **{item.get('action', '')}**: {item.get('description', '')}")
        else:
            lines.append("- Energy efficiency improvements across facilities")
            lines.append("- Transition to renewable electricity procurement")
            lines.append("- Supply chain engagement for Scope 3 reduction")
            lines.append("- Low-carbon transportation and logistics")
        return "\n".join(lines)

    def _md_offsetting(self, data):
        o = data.get("offsetting", {})
        return (f"## Our Offsetting Strategy\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Credits Retired | {_dec_comma(o.get('credits_tco2e', 0))} tCO2e |\n"
                f"| Registries Used | {', '.join(o.get('registries', ['Verra', 'Gold Standard']))} |\n"
                f"| Removal Share | {_pct(o.get('removal_pct', 0))} |\n"
                f"| ICVCM CCP Aligned | {_pct(o.get('ccp_aligned_pct', 0))} |\n"
                f"| SDG Contributions | {', '.join(str(s) for s in o.get('sdgs', [7, 13, 15]))} |\n\n"
                f"All credits are retired on recognized registries with unique serial "
                f"numbers and publicly available retirement certificates.")

    def _md_verification(self, data):
        v = data.get("verification", {})
        return (f"## Independent Verification\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| Verification Body | {v.get('body', 'N/A')} |\n"
                f"| Assurance Level | {v.get('level', 'Limited')} |\n"
                f"| Standard | {v.get('standard', 'ISO 14064-3:2019')} |\n"
                f"| Opinion | {v.get('opinion', 'Unmodified')} |\n"
                f"| Certificate | {v.get('certificate', 'N/A')} |")

    def _md_methodology(self, data):
        return ("## Methodology & Standards\n\n"
                "| Standard | Application |\n|----------|-------------|\n"
                "| GHG Protocol Corporate Standard | Emissions quantification boundary and methodology |\n"
                "| GHG Protocol Scope 3 Standard | Value chain emissions calculation |\n"
                "| PAS 2060:2014 | Carbon neutrality framework and declaration |\n"
                "| ISO 14064-1:2018 | GHG inventory quantification and reporting |\n"
                "| ISO 14064-3:2019 | Verification of GHG assertions |\n"
                "| ICVCM Core Carbon Principles | Carbon credit quality assessment |\n"
                "| VCMI Claims Code | Climate claim substantiation |")

    def _md_contact(self, data):
        contact = data.get("contact", {})
        return (f"## Contact & Further Information\n\n"
                f"For questions about this disclosure or our carbon neutrality programme:\n\n"
                f"- **Contact:** {contact.get('name', 'Sustainability Team')}\n"
                f"- **Email:** {contact.get('email', 'sustainability@company.com')}\n"
                f"- **Website:** {contact.get('website', '')}\n\n"
                f"Full methodology documentation and supporting evidence available upon request.")

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_header(self, data):
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        return f'<h1>Carbon Neutrality Disclosure<br><span style="font-size:0.6em;color:#689f38;">{org} | {year}</span></h1>'

    def _html_declaration(self, data):
        is_neutral = data.get("is_carbon_neutral", False)
        status = "Carbon Neutral" if is_neutral else "Working Towards Carbon Neutrality"
        total = data.get("total_emissions_tco2e", 0)
        return (f'<div class="declaration">'
                f'<div class="declaration-title">{status}</div>'
                f'<div class="declaration-status">{_dec_comma(total)} tCO2e quantified, reduced, and offset</div>'
                f'</div>')

    def _html_emissions(self, data):
        e = data.get("emissions", {})
        return (f'<h2>Emissions Profile</h2>'
                f'<div class="cards">'
                f'<div class="card"><div class="card-value">{_dec_comma(e.get("scope1", 0))}</div><div class="card-label">Scope 1 (tCO2e)</div></div>'
                f'<div class="card"><div class="card-value">{_dec_comma(e.get("scope2", 0))}</div><div class="card-label">Scope 2 (tCO2e)</div></div>'
                f'<div class="card"><div class="card-value">{_dec_comma(e.get("scope3", 0))}</div><div class="card-label">Scope 3 (tCO2e)</div></div>'
                f'</div>')

    def _html_offsetting(self, data):
        o = data.get("offsetting", {})
        return (f'<h2>Offsetting</h2>'
                f'<div class="cards">'
                f'<div class="card"><div class="card-value">{_dec_comma(o.get("credits_tco2e", 0))}</div><div class="card-label">Credits Retired (tCO2e)</div></div>'
                f'<div class="card"><div class="card-value">{_pct(o.get("removal_pct", 0))}</div><div class="card-label">Removal Share</div></div>'
                f'<div class="card"><div class="card-value">{_pct(o.get("ccp_aligned_pct", 0))}</div><div class="card-label">ICVCM CCP Aligned</div></div>'
                f'</div>')

    def _html_verification(self, data):
        v = data.get("verification", {})
        return (f'<h2>Verification</h2>'
                f'<table><tr><th>Field</th><th>Value</th></tr>'
                f'<tr><td>Body</td><td>{v.get("body", "N/A")}</td></tr>'
                f'<tr><td>Opinion</td><td>{v.get("opinion", "Unmodified")}</td></tr></table>')

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}</div>'
