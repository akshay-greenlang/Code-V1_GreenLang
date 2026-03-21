# -*- coding: utf-8 -*-
"""
SECClimateTemplate - SEC Climate Disclosure Template for PACK-030.

Renders SEC 10-K climate disclosure sections covering Reg S-K Item 1502-1506,
climate risk in business description (Item 1), climate risks in risk factors
(Item 1A), climate impact in MD&A (Item 7), Scope 1/2 emissions with XBRL/
iXBRL tagging, attestation requirements, and SOX compliance notes.
Multi-format output (MD, HTML, JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Item 1: Climate Risks in Business Description
    3.  Item 1A: Climate Risk Factors
    4.  Item 7: Climate Impacts in MD&A
    5.  Reg S-K 1502: GHG Emissions (Scope 1 & 2)
    6.  Reg S-K 1503: Climate-Related Targets
    7.  Reg S-K 1504: Transition Plan
    8.  Reg S-K 1505: Scenario Analysis
    9.  Attestation Requirements
    10. XBRL/iXBRL Tagging
    11. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "sec_climate"
_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"

SEC_XBRL_TAXONOMY = {
    "scope1": {"element": "us-gaap:GreenHouseGasEmissionsDirectScope1", "type": "monetary", "period": "duration"},
    "scope2": {"element": "us-gaap:GreenHouseGasEmissionsIndirectScope2", "type": "monetary", "period": "duration"},
    "scope1_intensity": {"element": "us-gaap:GreenHouseGasEmissionsIntensityScope1", "type": "perShare", "period": "duration"},
    "target_year": {"element": "us-gaap:ClimateTargetYear", "type": "date", "period": "instant"},
    "target_reduction": {"element": "us-gaap:ClimateTargetReductionPercentage", "type": "percent", "period": "instant"},
}

XBRL_TAGS: Dict[str, str] = {
    "scope1": "gl:SECScope1Emissions", "scope2": "gl:SECScope2Emissions",
    "total_s12": "gl:SECScope12Total", "intensity": "gl:SECEmissionsIntensity",
    "attestation_level": "gl:SECAttestationLevel", "target_year": "gl:SECTargetYear",
}

def _utcnow(): return datetime.now(timezone.utc).replace(microsecond=0)
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


class SECClimateTemplate:
    """SEC 10-K climate disclosure template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_item1(data), self._md_item1a(data), self._md_item7(data),
            self._md_1502_emissions(data), self._md_1503_targets(data),
            self._md_1504_transition(data), self._md_1505_scenario(data),
            self._md_attestation(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [self._html_header(data), self._html_executive_summary(data),
                 self._html_1502_emissions(data), self._html_1503_targets(data),
                 self._html_attestation(data), self._html_xbrl(data),
                 self._html_audit(data), self._html_footer(data)]
        body = "\n".join(parts)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>SEC Climate - {data.get("org_name","")}</title>\n<style>\n{css}\n</style>\n</head>\n'
                f'<body>\n<div class="report">\n{body}\n</div>\n<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        s1 = float(data.get("scope1",0)); s2 = float(data.get("scope2",0))
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "fiscal_year": data.get("fiscal_year",""),
            "framework": "SEC", "filing_type": "10-K",
            "emissions": {"scope1": str(s1), "scope2": str(s2), "total": str(s1+s2)},
            "targets": data.get("targets", []),
            "attestation": data.get("attestation", {}),
            "xbrl_taxonomy": SEC_XBRL_TAXONOMY, "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"SEC Climate - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# SEC Climate Disclosure (10-K)\n\n**Registrant:** {data.get('org_name','')}  \n**Fiscal Year:** {data.get('fiscal_year','')}  \n**Filing Type:** 10-K (Annual Report)  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0))
        att = data.get("attestation",{})
        lines = ["## 1. Executive Summary\n", "| Metric | Value |", "|--------|-------|",
                  f"| Scope 1 | {_dec_comma(s1,0)} tCO2e |", f"| Scope 2 | {_dec_comma(s2,0)} tCO2e |",
                  f"| Total (S1+S2) | {_dec_comma(s1+s2,0)} tCO2e |",
                  f"| Attestation Level | {att.get('level','Limited assurance')} |",
                  f"| Targets Set | {len(data.get('targets',[]))} |",
                  f"| XBRL Tagging | Required |"]
        return "\n".join(lines)

    def _md_item1(self, data):
        item1 = data.get("item1", {})
        lines = ["## 2. Item 1: Climate in Business Description\n",
                  f"{item1.get('narrative', 'The Company has identified climate change as a key factor affecting its operations and long-term strategy.')}\n",
                  "| Parameter | Value |", "|-----------|-------|",
                  f"| Climate-Sensitive Revenue | {item1.get('climate_sensitive_revenue','')} |",
                  f"| Physical Risk Exposure | {item1.get('physical_exposure','')} |",
                  f"| Transition Risk Exposure | {item1.get('transition_exposure','')} |"]
        return "\n".join(lines)

    def _md_item1a(self, data):
        risks = data.get("risk_factors", [])
        lines = ["## 3. Item 1A: Climate Risk Factors\n",
                  "| # | Risk Factor | Category | Financial Impact |",
                  "|---|------------ |----------|------------------|"]
        for i, r in enumerate(risks, 1):
            lines.append(f"| {i} | {r.get('name','')} | {r.get('category','')} | {r.get('financial_impact','')} |")
        if not risks: lines.append("| - | _No climate risk factors_ | - | - |")
        return "\n".join(lines)

    def _md_item7(self, data):
        mda = data.get("mda", {})
        lines = ["## 4. Item 7: Climate Impacts in MD&A\n",
                  f"{mda.get('narrative', 'Climate-related factors have had the following impacts on results of operations and financial condition.')}\n",
                  "| Impact Area | Description | Financial Effect |", "|-------------|-------------|------------------|"]
        for impact in mda.get("impacts", []):
            lines.append(f"| {impact.get('area','')} | {impact.get('description','')} | {impact.get('financial_effect','')} |")
        return "\n".join(lines)

    def _md_1502_emissions(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0))
        revenue = float(data.get("revenue",1))
        intensity = (s1+s2)/revenue*1000000 if revenue else 0
        lines = ["## 5. Reg S-K 1502: GHG Emissions\n",
                  "| Metric | Value | XBRL Element |", "|--------|-------|--------------|",
                  f"| Scope 1 | {_dec_comma(s1,0)} tCO2e | {SEC_XBRL_TAXONOMY['scope1']['element']} |",
                  f"| Scope 2 | {_dec_comma(s2,0)} tCO2e | {SEC_XBRL_TAXONOMY['scope2']['element']} |",
                  f"| Total | {_dec_comma(s1+s2,0)} tCO2e | - |",
                  f"| Intensity | {_dec(intensity)} tCO2e/$M revenue | {SEC_XBRL_TAXONOMY['scope1_intensity']['element']} |"]
        return "\n".join(lines)

    def _md_1503_targets(self, data):
        targets = data.get("targets", [])
        lines = ["## 6. Reg S-K 1503: Climate Targets\n",
                  "| # | Target | Scope | Base Year | Target Year | Reduction |",
                  "|---|--------|-------|:---------:|:-----------:|----------:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {t.get('scope','')} | {t.get('base_year','')} | {t.get('target_year','')} | {_dec(t.get('reduction_pct',0))}% |")
        if not targets: lines.append("| - | _No targets_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_1504_transition(self, data):
        plan = data.get("transition_plan", {})
        lines = ["## 7. Reg S-K 1504: Transition Plan\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Status | {plan.get('status','Published')} |",
                  f"| Net-Zero Year | {plan.get('net_zero_year','')} |",
                  f"| Key Actions | {plan.get('key_actions','')} |",
                  f"| Capital Allocation | {plan.get('capex','')} |"]
        return "\n".join(lines)

    def _md_1505_scenario(self, data):
        scenario = data.get("scenario_analysis", {})
        lines = ["## 8. Reg S-K 1505: Scenario Analysis\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Scenarios Used | {scenario.get('scenarios','1.5C, 2C, 4C')} |",
                  f"| Time Horizons | {scenario.get('horizons','2030, 2050')} |",
                  f"| Financial Impact | {scenario.get('financial_impact','')} |",
                  f"| Resilience Assessment | {scenario.get('resilience','')} |"]
        return "\n".join(lines)

    def _md_attestation(self, data):
        att = data.get("attestation", {})
        lines = ["## 9. Attestation Requirements\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Required | {att.get('required','Yes (LAF)')} |",
                  f"| Level | {att.get('level','Limited assurance')} |",
                  f"| Provider | {att.get('provider','')} |",
                  f"| Standard | {att.get('standard','PCAOB AS 3000')} |",
                  f"| Scope | {att.get('scope','Scope 1 + 2 emissions')} |",
                  f"| Opinion | {att.get('opinion','')} |"]
        return "\n".join(lines)

    def _md_xbrl(self, data):
        lines = ["## 10. XBRL/iXBRL Tagging\n",
                  "| Element | Type | Period | Value |", "|---------|------|--------|-------|"]
        for key, info in SEC_XBRL_TAXONOMY.items():
            lines.append(f"| {info['element']} | {info['type']} | {info['period']} | - |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f"## 11. Audit Trail & Provenance\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*SEC 10-K climate disclosure.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#f1f8e9;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>SEC Climate Disclosure (10-K)</h1>\n<p><strong>{data.get("org_name","")}</strong> | FY {data.get("fiscal_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0))
        return (f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(s1,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(s2,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(s1+s2,0)}</div><div class="card-unit">tCO2e</div></div>\n</div>')

    def _html_1502_emissions(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0))
        return (f'<h2>2. Reg S-K 1502</h2>\n<table>\n<tr><th>Scope</th><th>Emissions</th><th>XBRL</th></tr>\n'
                f'<tr><td>Scope 1</td><td>{_dec_comma(s1,0)}</td><td><code>{SEC_XBRL_TAXONOMY["scope1"]["element"]}</code></td></tr>\n'
                f'<tr><td>Scope 2</td><td>{_dec_comma(s2,0)}</td><td><code>{SEC_XBRL_TAXONOMY["scope2"]["element"]}</code></td></tr>\n</table>')

    def _html_1503_targets(self, data):
        targets = data.get("targets",[]); rows = ""
        for i, t in enumerate(targets, 1):
            rows += f'<tr><td>{i}</td><td>{t.get("name","")}</td><td>{t.get("target_year","")}</td><td>{_dec(t.get("reduction_pct",0))}%</td></tr>\n'
        return f'<h2>3. Targets</h2>\n<table>\n<tr><th>#</th><th>Target</th><th>Year</th><th>Reduction</th></tr>\n{rows}</table>'

    def _html_attestation(self, data):
        att = data.get("attestation",{})
        return (f'<h2>4. Attestation</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n'
                f'<tr><td>Level</td><td>{att.get("level","Limited")}</td></tr>\n'
                f'<tr><td>Provider</td><td>{att.get("provider","")}</td></tr>\n</table>')

    def _html_xbrl(self, data):
        rows = ""
        for key, info in SEC_XBRL_TAXONOMY.items():
            rows += f'<tr><td>{key}</td><td><code>{info["element"]}</code></td><td>{info["type"]}</td></tr>\n'
        return f'<h2>5. XBRL</h2>\n<table>\n<tr><th>Key</th><th>Element</th><th>Type</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>6. Audit</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n<tr><td>ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - SEC Climate</div>'
