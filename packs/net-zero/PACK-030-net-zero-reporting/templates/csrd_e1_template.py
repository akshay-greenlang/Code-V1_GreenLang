# -*- coding: utf-8 -*-
"""
CSRDE1Template - CSRD ESRS E1 Climate Change Template for PACK-030.

Renders CSRD ESRS E1 Climate Change disclosure covering all 9 disclosure
requirements (E1-1 through E1-9): transition plan, policies, actions,
targets, energy consumption, emissions, removals/credits, internal
carbon pricing, and anticipated financial effects. Includes digital
taxonomy tagging and CSRD compliance scoring. Multi-format output
(MD, HTML, JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  E1-1: Transition Plan for Climate Change Mitigation
    3.  E1-2: Policies Related to Climate Change
    4.  E1-3: Actions and Resources
    5.  E1-4: GHG Emission Reduction Targets
    6.  E1-5: Energy Consumption and Mix
    7.  E1-6: Gross Scopes 1, 2, 3 Emissions
    8.  E1-7: GHG Removals and Carbon Credits
    9.  E1-8: Internal Carbon Pricing
    10. E1-9: Anticipated Financial Effects
    11. CSRD Compliance Scoring
    12. Digital Taxonomy Tags
    13. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "csrd_e1"
_PRIMARY = "#003399"
_SECONDARY = "#0055a4"
_ACCENT = "#ffcc00"
_LIGHT = "#e3ecf9"
_LIGHTER = "#f0f5ff"

ESRS_E1_DISCLOSURES = [
    {"code": "E1-1", "title": "Transition plan for climate change mitigation", "type": "Narrative + Quantitative"},
    {"code": "E1-2", "title": "Policies related to climate change mitigation and adaptation", "type": "Narrative"},
    {"code": "E1-3", "title": "Actions and resources in relation to climate change policies", "type": "Narrative + Quantitative"},
    {"code": "E1-4", "title": "Targets related to climate change mitigation and adaptation", "type": "Quantitative"},
    {"code": "E1-5", "title": "Energy consumption and mix", "type": "Quantitative"},
    {"code": "E1-6", "title": "Gross Scopes 1, 2 and 3 GHG emissions", "type": "Quantitative"},
    {"code": "E1-7", "title": "GHG removals and GHG mitigation projects financed through carbon credits", "type": "Quantitative"},
    {"code": "E1-8", "title": "Internal carbon pricing", "type": "Quantitative"},
    {"code": "E1-9", "title": "Anticipated financial effects from material physical and transition risks", "type": "Quantitative"},
]

XBRL_TAGS: Dict[str, str] = {
    "scope1": "esrs:E1_6_Scope1", "scope2_location": "esrs:E1_6_Scope2Location",
    "scope2_market": "esrs:E1_6_Scope2Market", "scope3": "esrs:E1_6_Scope3",
    "energy_total": "esrs:E1_5_EnergyTotal", "renewable_pct": "esrs:E1_5_RenewablePct",
    "carbon_credits": "esrs:E1_7_CarbonCredits", "internal_carbon_price": "esrs:E1_8_CarbonPrice",
    "transition_plan": "esrs:E1_1_TransitionPlan", "target_year": "esrs:E1_4_TargetYear",
}

def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(data):
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def _dec(val, places=2):
    try: return str(Decimal(str(val)).quantize(Decimal("0." + "0" * places), rounding=ROUND_HALF_UP))
    except: return str(val)
def _dec_comma(val, places=2):
    try:
        rounded = Decimal(str(val)).quantize(Decimal("0." + "0" * places if places > 0 else "0"), rounding=ROUND_HALF_UP)
        parts = str(rounded).split("."); ip = parts[0]; neg = ip.startswith("-")
        if neg: ip = ip[1:]
        fmt = ""
        for i, c in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = c + fmt
        if neg: fmt = "-" + fmt
        return fmt + ("." + parts[1] if len(parts) > 1 else "")
    except: return str(val)

class CSRDE1Template:
    """CSRD ESRS E1 Climate Change template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_e1_1(data), self._md_e1_2(data), self._md_e1_3(data),
            self._md_e1_4(data), self._md_e1_5(data), self._md_e1_6(data),
            self._md_e1_7(data), self._md_e1_8(data), self._md_e1_9(data),
            self._md_compliance(data), self._md_taxonomy(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        parts = [self._html_header(data), self._html_executive_summary(data),
                 self._html_e1_6(data), self._html_e1_5(data),
                 self._html_compliance(data), self._html_taxonomy(data),
                 self._html_audit(data), self._html_footer(data)]
        body = "\n".join(parts)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ESRS E1 - {data.get("org_name","")}</title>\n<style>\n{css}\n</style>\n</head>\n'
                f'<body>\n<div class="report">\n{body}\n</div>\n<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        compliance = self._calculate_compliance(data)
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "reporting_year": data.get("reporting_year",""),
            "framework": "CSRD", "standard": "ESRS E1",
            "disclosures": {d["code"]: data.get(d["code"].lower().replace("-","_"), {}) for d in ESRS_E1_DISCLOSURES},
            "compliance": compliance,
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"ESRS E1 - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _calculate_compliance(self, data):
        checks = {}
        for d in ESRS_E1_DISCLOSURES:
            key = d["code"].lower().replace("-","_")
            checks[d["code"]] = bool(data.get(key)) or bool(data.get(key, {}).get("disclosed"))
        passed = sum(1 for v in checks.values() if v); total = len(checks)
        return {"checks": checks, "passed": passed, "total": total, "score": round(passed/total*100, 1) if total else 0}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# ESRS E1 Climate Change Disclosure\n\n**Organization:** {data.get('org_name','')}  \n**Reporting Year:** {data.get('reporting_year','')}  \n**Standard:** ESRS E1 (CSRD)  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        comp = self._calculate_compliance(data)
        lines = ["## 1. Executive Summary\n", "| Metric | Value |", "|--------|-------|",
                  f"| Scope 1 | {_dec_comma(s1,0)} tCO2e |", f"| Scope 2 | {_dec_comma(s2,0)} tCO2e |",
                  f"| Scope 3 | {_dec_comma(s3,0)} tCO2e |", f"| Total | {_dec_comma(s1+s2+s3,0)} tCO2e |",
                  f"| ESRS E1 Compliance | {comp['score']}% ({comp['passed']}/{comp['total']}) |",
                  f"| Transition Plan | {data.get('e1_1',{}).get('status','Published')} |"]
        return "\n".join(lines)

    def _md_e1_1(self, data):
        e1 = data.get("e1_1", {})
        lines = ["## 2. E1-1: Transition Plan\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Status | {e1.get('status','Published')} |", f"| Net-Zero Year | {e1.get('net_zero_year','')} |",
                  f"| Approved By | {e1.get('approved_by','Board of Directors')} |",
                  f"| Key Actions | {e1.get('key_actions','')} |",
                  f"| CapEx Allocated | {e1.get('capex','')} |",
                  f"| Locked-in Emissions | {e1.get('locked_in','')} |"]
        return "\n".join(lines)

    def _md_e1_2(self, data):
        e1 = data.get("e1_2", {})
        policies = e1.get("policies", [])
        lines = ["## 3. E1-2: Policies Related to Climate Change\n",
                  "| # | Policy | Scope | Status |", "|---|--------|-------|--------|"]
        for i, p in enumerate(policies, 1):
            lines.append(f"| {i} | {p.get('name','')} | {p.get('scope','')} | {p.get('status','Active')} |")
        if not policies: lines.append("| - | _No policies disclosed_ | - | - |")
        return "\n".join(lines)

    def _md_e1_3(self, data):
        e1 = data.get("e1_3", {})
        actions = e1.get("actions", [])
        lines = ["## 4. E1-3: Actions and Resources\n",
                  "| # | Action | Investment | Expected Reduction | Timeline |",
                  "|---|--------|-----------|-------------------:|----------|"]
        for i, a in enumerate(actions, 1):
            lines.append(f"| {i} | {a.get('name','')} | {a.get('investment','')} | {_dec_comma(a.get('reduction',0),0)} tCO2e | {a.get('timeline','')} |")
        if not actions: lines.append("| - | _No actions disclosed_ | - | - | - |")
        return "\n".join(lines)

    def _md_e1_4(self, data):
        e1 = data.get("e1_4", {})
        targets = e1.get("targets", [])
        lines = ["## 5. E1-4: GHG Emission Reduction Targets\n",
                  "| # | Target | Scope | Base Year | Target Year | Reduction (%) | SBTi |",
                  "|---|--------|-------|:---------:|:-----------:|--------------:|:----:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {t.get('scope','')} | {t.get('base_year','')} | {t.get('target_year','')} | {_dec(t.get('reduction_pct',0))}% | {t.get('sbti','No')} |")
        if not targets: lines.append("| - | _No targets_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_e1_5(self, data):
        e1 = data.get("e1_5", {})
        lines = ["## 6. E1-5: Energy Consumption and Mix\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Total Energy | {_dec_comma(e1.get('total_mwh',0),0)} MWh |",
                  f"| Renewable Energy | {_dec_comma(e1.get('renewable_mwh',0),0)} MWh ({_dec(e1.get('renewable_pct',0))}%) |",
                  f"| Non-Renewable | {_dec_comma(e1.get('non_renewable_mwh',0),0)} MWh |",
                  f"| Energy from Coal | {_dec_comma(e1.get('coal_mwh',0),0)} MWh |",
                  f"| Energy from Gas | {_dec_comma(e1.get('gas_mwh',0),0)} MWh |",
                  f"| Energy Intensity | {_dec(e1.get('intensity',0))} MWh/revenue unit |"]
        return "\n".join(lines)

    def _md_e1_6(self, data):
        s1=float(data.get("scope1",0)); s2l=float(data.get("scope2_location",0))
        s2m=float(data.get("scope2_market",0)); s3=float(data.get("scope3",0))
        total = s1+s2l+s3
        lines = ["## 7. E1-6: Gross Scopes 1, 2, 3 GHG Emissions\n",
                  "| Scope | Method | Emissions (tCO2e) | Share (%) |", "|-------|--------|------------------:|----------:|",
                  f"| Scope 1 | Direct | {_dec_comma(s1,0)} | {_dec(s1/total*100 if total else 0)}% |",
                  f"| Scope 2 | Location | {_dec_comma(s2l,0)} | {_dec(s2l/total*100 if total else 0)}% |",
                  f"| Scope 2 | Market | {_dec_comma(s2m,0)} | - |",
                  f"| Scope 3 | Indirect | {_dec_comma(s3,0)} | {_dec(s3/total*100 if total else 0)}% |",
                  f"| **Total** | | **{_dec_comma(total,0)}** | **100%** |"]
        return "\n".join(lines)

    def _md_e1_7(self, data):
        e1 = data.get("e1_7", {})
        lines = ["## 8. E1-7: GHG Removals and Carbon Credits\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| GHG Removals | {_dec_comma(e1.get('removals',0),0)} tCO2e |",
                  f"| Carbon Credits Purchased | {_dec_comma(e1.get('credits_purchased',0),0)} tCO2e |",
                  f"| Carbon Credits Retired | {_dec_comma(e1.get('credits_retired',0),0)} tCO2e |",
                  f"| Credit Type | {e1.get('credit_type','Nature-based + technology-based')} |",
                  f"| Certification Standard | {e1.get('certification','Verra VCS, Gold Standard')} |"]
        return "\n".join(lines)

    def _md_e1_8(self, data):
        e1 = data.get("e1_8", {})
        lines = ["## 9. E1-8: Internal Carbon Pricing\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Applied | {e1.get('applied','Yes')} |", f"| Price | {e1.get('price','EUR 100/tCO2e')} |",
                  f"| Type | {e1.get('type','Shadow price')} |",
                  f"| Application | {e1.get('application','CapEx decisions, R&D priorities')} |"]
        return "\n".join(lines)

    def _md_e1_9(self, data):
        e1 = data.get("e1_9", {})
        effects = e1.get("financial_effects", [])
        lines = ["## 10. E1-9: Anticipated Financial Effects\n",
                  "| # | Risk/Opportunity | Type | Time Horizon | Financial Impact |",
                  "|---|-----------------|------|:------------:|------------------|"]
        for i, e in enumerate(effects, 1):
            lines.append(f"| {i} | {e.get('name','')} | {e.get('type','')} | {e.get('horizon','')} | {e.get('impact','')} |")
        if not effects: lines.append("| - | _No financial effects_ | - | - | - |")
        return "\n".join(lines)

    def _md_compliance(self, data):
        comp = self._calculate_compliance(data)
        lines = ["## 11. CSRD Compliance Scoring\n", f"**Score:** {comp['score']}% ({comp['passed']}/{comp['total']})\n",
                  "| Code | Disclosure | Status |", "|------|-----------|--------|"]
        for d in ESRS_E1_DISCLOSURES:
            status = "Disclosed" if comp["checks"].get(d["code"]) else "Missing"
            lines.append(f"| {d['code']} | {d['title']} | {status} |")
        return "\n".join(lines)

    def _md_taxonomy(self, data):
        lines = ["## 12. Digital Taxonomy Tags\n", "| Data Point | ESRS Tag |", "|------------|----------|"]
        for key, tag in XBRL_TAGS.items(): lines.append(f"| {key.replace('_',' ').title()} | {tag} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f"## 13. Audit Trail & Provenance\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*CSRD ESRS E1 climate change disclosure.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_ACCENT};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #bbdefb;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#e8eaf6;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_ACCENT};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".card-unit{{font-size:0.75em;color:{_SECONDARY};}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_SECONDARY};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>ESRS E1 Climate Change</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        comp = self._calculate_compliance(data)
        return (f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(s1,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(s2,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 3</div><div class="card-value">{_dec_comma(s3,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Compliance</div><div class="card-value">{comp["score"]}%</div></div>\n</div>')

    def _html_e1_6(self, data):
        s1=float(data.get("scope1",0)); s2l=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        return (f'<h2>2. E1-6 Emissions</h2>\n<table>\n<tr><th>Scope</th><th>Emissions</th></tr>\n'
                f'<tr><td>Scope 1</td><td>{_dec_comma(s1,0)}</td></tr>\n<tr><td>Scope 2</td><td>{_dec_comma(s2l,0)}</td></tr>\n'
                f'<tr><td>Scope 3</td><td>{_dec_comma(s3,0)}</td></tr>\n<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2l+s3,0)}</strong></td></tr>\n</table>')

    def _html_e1_5(self, data):
        e1 = data.get("e1_5",{})
        return (f'<h2>3. E1-5 Energy</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n'
                f'<tr><td>Total</td><td>{_dec_comma(e1.get("total_mwh",0),0)} MWh</td></tr>\n'
                f'<tr><td>Renewable %</td><td>{_dec(e1.get("renewable_pct",0))}%</td></tr>\n</table>')

    def _html_compliance(self, data):
        comp = self._calculate_compliance(data); rows = ""
        for d in ESRS_E1_DISCLOSURES:
            s = "Disclosed" if comp["checks"].get(d["code"]) else "Missing"
            rows += f'<tr><td>{d["code"]}</td><td>{d["title"]}</td><td>{s}</td></tr>\n'
        return f'<h2>4. Compliance</h2>\n<p>Score: {comp["score"]}%</p>\n<table>\n<tr><th>Code</th><th>Disclosure</th><th>Status</th></tr>\n{rows}</table>'

    def _html_taxonomy(self, data):
        rows = ""
        for key, tag in XBRL_TAGS.items(): rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>5. Taxonomy</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>6. Audit</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n<tr><td>ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - ESRS E1</div>'
