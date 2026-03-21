# -*- coding: utf-8 -*-
"""
ISSBS2Template - ISSB IFRS S2 Climate Disclosure Template for PACK-030.

Renders IFRS S2 Climate-related Disclosures compliant report covering
governance (para 5-7), strategy (para 8-22), risk management (para
23-27), metrics & targets (para 28-37), industry-specific SASB metrics,
and XBRL digital reporting tags. Multi-format output (MD, HTML, JSON,
PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Governance (S2 para 5-7)
    3.  Strategy (S2 para 8-22)
    4.  Risk Management (S2 para 23-27)
    5.  Metrics & Targets (S2 para 28-37)
    6.  Cross-Industry Metrics (Scope 1/2/3)
    7.  Industry-Specific Metrics (SASB)
    8.  Transition Plan Disclosure
    9.  Climate Resilience Assessment
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "issb_s2"
_PRIMARY = "#0d47a1"
_SECONDARY = "#1565c0"
_ACCENT = "#42a5f5"
_LIGHT = "#e3f2fd"
_LIGHTER = "#f5f9ff"

ISSB_S2_PARAGRAPHS = {
    "governance": ["5: Board oversight", "6: Management role", "7: Climate competence"],
    "strategy": ["8-9: Climate risks/opportunities", "10-12: Business model impact", "13-15: Scenario analysis", "16-19: Transition plan", "20-22: Financial effects"],
    "risk_management": ["23: Identification process", "24: Assessment process", "25: Management process", "26: ERM integration", "27: Value chain risks"],
    "metrics": ["28: Cross-industry metrics", "29: Scope 1/2/3 emissions", "30-33: Industry metrics", "34: Climate targets", "35-37: Carbon credits/offsets"],
}

XBRL_TAGS: Dict[str, str] = {
    "scope1": "gl:ISSBS2Scope1", "scope2": "gl:ISSBS2Scope2", "scope3": "gl:ISSBS2Scope3",
    "financed_emissions": "gl:ISSBS2FinancedEmissions",
    "transition_plan": "gl:ISSBS2TransitionPlan", "resilience_assessment": "gl:ISSBS2Resilience",
    "carbon_credits": "gl:ISSBS2CarbonCredits", "internal_carbon_price": "gl:ISSBS2InternalCarbonPrice",
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
        parts = str(rounded).split("."); int_part = parts[0]; neg = int_part.startswith("-")
        if neg: int_part = int_part[1:]
        fmt = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = ch + fmt
        if neg: fmt = "-" + fmt
        return fmt + ("." + parts[1] if len(parts) > 1 else "")
    except: return str(val)


class ISSBS2Template:
    """ISSB IFRS S2 template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_governance(data), self._md_strategy(data),
            self._md_risk_management(data), self._md_metrics_targets(data),
            self._md_cross_industry(data), self._md_industry_specific(data),
            self._md_transition_plan(data), self._md_resilience(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [self._html_header(data), self._html_executive_summary(data),
                 self._html_cross_industry(data), self._html_industry_specific(data),
                 self._html_xbrl(data), self._html_audit(data), self._html_footer(data)]
        body = "\n".join(parts)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ISSB S2 - {data.get("org_name","")}</title>\n<style>\n{css}\n</style>\n</head>\n'
                f'<body>\n<div class="report">\n{body}\n</div>\n<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "reporting_year": data.get("reporting_year",""),
            "framework": "ISSB", "standard": "IFRS S2",
            "governance": data.get("governance", {}), "strategy": data.get("strategy", {}),
            "risk_management": data.get("risk_management", {}),
            "emissions": {"scope1": str(data.get("scope1",0)), "scope2": str(data.get("scope2",0)), "scope3": str(data.get("scope3",0))},
            "industry_metrics": data.get("industry_metrics", []),
            "targets": data.get("targets", []),
            "transition_plan": data.get("transition_plan", {}),
            "carbon_credits": data.get("carbon_credits", {}),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"ISSB S2 - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# IFRS S2 Climate-related Disclosures\n\n**Organization:** {data.get('org_name','')}  \n**Reporting Period:** {data.get('reporting_year','')}  \n**Standard:** IFRS S2 (ISSB)  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0)); s3=float(data.get("scope3",0))
        lines = ["## 1. Executive Summary\n", "| Metric | Value |", "|--------|-------|",
                  f"| Scope 1 | {_dec_comma(s1,0)} tCO2e |", f"| Scope 2 | {_dec_comma(s2,0)} tCO2e |",
                  f"| Scope 3 | {_dec_comma(s3,0)} tCO2e |", f"| Total | {_dec_comma(s1+s2+s3,0)} tCO2e |",
                  f"| Industry Metrics | {len(data.get('industry_metrics',[]))} |",
                  f"| Targets | {len(data.get('targets',[]))} |",
                  f"| Transition Plan | {data.get('transition_plan',{}).get('status','Published')} |"]
        return "\n".join(lines)

    def _md_governance(self, data):
        gov = data.get("governance", {})
        lines = ["## 2. Governance (S2 para 5-7)\n", "| Requirement | Disclosure |", "|-------------|-----------|"]
        for para in ISSB_S2_PARAGRAPHS["governance"]:
            key = para.split(":")[0].strip().replace("-","_").replace(" ","_")
            lines.append(f"| {para} | {gov.get(key, 'See full report')} |")
        return "\n".join(lines)

    def _md_strategy(self, data):
        strat = data.get("strategy", {})
        lines = ["## 3. Strategy (S2 para 8-22)\n", "| Requirement | Disclosure |", "|-------------|-----------|"]
        for para in ISSB_S2_PARAGRAPHS["strategy"]:
            key = para.split(":")[0].strip().replace("-","_").replace(" ","_")
            lines.append(f"| {para} | {strat.get(key, 'See full report')} |")
        return "\n".join(lines)

    def _md_risk_management(self, data):
        rm = data.get("risk_management", {})
        lines = ["## 4. Risk Management (S2 para 23-27)\n", "| Requirement | Disclosure |", "|-------------|-----------|"]
        for para in ISSB_S2_PARAGRAPHS["risk_management"]:
            key = para.split(":")[0].strip().replace("-","_").replace(" ","_")
            lines.append(f"| {para} | {rm.get(key, 'See full report')} |")
        return "\n".join(lines)

    def _md_metrics_targets(self, data):
        targets = data.get("targets", [])
        lines = ["## 5. Metrics & Targets (S2 para 28-37)\n",
                  "| # | Target | Scope | Base Year | Target Year | Reduction (%) | Progress (%) |",
                  "|---|--------|-------|:---------:|:-----------:|--------------:|-------------:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {t.get('scope','')} | {t.get('base_year','')} | {t.get('target_year','')} | {_dec(t.get('reduction_pct',0))}% | {_dec(t.get('progress_pct',0))}% |")
        if not targets: lines.append("| - | _No targets_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_cross_industry(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0)); s3=float(data.get("scope3",0))
        total = s1+s2+s3
        lines = ["## 6. Cross-Industry Metrics (S2 para 29)\n",
                  "| Scope | Emissions (tCO2e) | Share (%) |", "|-------|------------------:|----------:|",
                  f"| Scope 1 | {_dec_comma(s1,0)} | {_dec(s1/total*100 if total else 0)}% |",
                  f"| Scope 2 | {_dec_comma(s2,0)} | {_dec(s2/total*100 if total else 0)}% |",
                  f"| Scope 3 | {_dec_comma(s3,0)} | {_dec(s3/total*100 if total else 0)}% |",
                  f"| **Total** | **{_dec_comma(total,0)}** | **100%** |"]
        return "\n".join(lines)

    def _md_industry_specific(self, data):
        metrics = data.get("industry_metrics", [])
        lines = ["## 7. Industry-Specific Metrics (SASB-aligned)\n",
                  "| # | Metric | SASB Code | Value | Unit |", "|---|--------|-----------|-------|------|"]
        for i, m in enumerate(metrics, 1):
            lines.append(f"| {i} | {m.get('name','')} | {m.get('sasb_code','')} | {m.get('value','')} | {m.get('unit','')} |")
        if not metrics: lines.append("| - | _No industry metrics_ | - | - | - |")
        return "\n".join(lines)

    def _md_transition_plan(self, data):
        plan = data.get("transition_plan", {})
        lines = ["## 8. Transition Plan Disclosure (S2 para 16-19)\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Status | {plan.get('status','Published')} |", f"| Net-Zero Year | {plan.get('net_zero_year','')} |",
                  f"| Key Actions | {plan.get('key_actions','')} |", f"| Investment | {plan.get('investment','')} |",
                  f"| Dependencies | {plan.get('dependencies','')} |"]
        return "\n".join(lines)

    def _md_resilience(self, data):
        res = data.get("resilience", {})
        lines = ["## 9. Climate Resilience Assessment (S2 para 22)\n", "| Dimension | Assessment |", "|-----------|-----------|",
                  f"| Physical Resilience | {res.get('physical','Assessed')} |",
                  f"| Transition Resilience | {res.get('transition','Assessed')} |",
                  f"| Scenario Analysis | {res.get('scenario_analysis','1.5C, 2C, 4C')} |",
                  f"| Overall | {res.get('overall','Moderate')} |"]
        return "\n".join(lines)

    def _md_xbrl(self, data):
        lines = ["## 10. XBRL Tagging Summary\n", "| Data Point | XBRL Tag |", "|------------|----------|"]
        for key, tag in XBRL_TAGS.items(): lines.append(f"| {key.replace('_',' ').title()} | {tag} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f"## 11. Audit Trail & Provenance\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*IFRS S2 climate-related disclosure.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #bbdefb;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#e8eaf6;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>IFRS S2 Climate Disclosure</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0)); s3=float(data.get("scope3",0))
        return (f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(s1,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(s2,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 3</div><div class="card-value">{_dec_comma(s3,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(s1+s2+s3,0)}</div><div class="card-unit">tCO2e</div></div>\n</div>')

    def _html_cross_industry(self, data):
        s1=float(data.get("scope1",0)); s2=float(data.get("scope2",0)); s3=float(data.get("scope3",0))
        return (f'<h2>2. Cross-Industry Metrics</h2>\n<table>\n<tr><th>Scope</th><th>Emissions</th></tr>\n'
                f'<tr><td>Scope 1</td><td>{_dec_comma(s1,0)}</td></tr>\n<tr><td>Scope 2</td><td>{_dec_comma(s2,0)}</td></tr>\n'
                f'<tr><td>Scope 3</td><td>{_dec_comma(s3,0)}</td></tr>\n<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2+s3,0)}</strong></td></tr>\n</table>')

    def _html_industry_specific(self, data):
        metrics = data.get("industry_metrics", []); rows = ""
        for i, m in enumerate(metrics, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("name","")}</td><td>{m.get("sasb_code","")}</td><td>{m.get("value","")}</td></tr>\n'
        return f'<h2>3. Industry Metrics</h2>\n<table>\n<tr><th>#</th><th>Metric</th><th>SASB</th><th>Value</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data):
        rows = ""
        for key, tag in XBRL_TAGS.items(): rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>4. XBRL</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>5. Audit</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n<tr><td>ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - ISSB S2</div>'
