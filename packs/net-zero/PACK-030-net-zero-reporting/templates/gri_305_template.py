# -*- coding: utf-8 -*-
"""
GRI305Template - GRI 305 Emissions Disclosure Template for PACK-030.

Renders GRI 305 emissions disclosures covering 305-1 (Direct Scope 1),
305-2 (Indirect Scope 2), 305-3 (Other Indirect Scope 3), 305-4
(Intensity), 305-5 (Reduction), 305-6 (ODS), 305-7 (NOx/SOx), and
GRI Content Index table. Multi-format output (MD, HTML, JSON, PDF)
with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  GRI 305-1: Direct (Scope 1) GHG Emissions
    3.  GRI 305-2: Energy Indirect (Scope 2) GHG Emissions
    4.  GRI 305-3: Other Indirect (Scope 3) GHG Emissions
    5.  GRI 305-4: GHG Emissions Intensity
    6.  GRI 305-5: Reduction of GHG Emissions
    7.  GRI 305-6: Emissions of Ozone-Depleting Substances
    8.  GRI 305-7: NOx, SOx, and Other Significant Air Emissions
    9.  GRI Content Index
    10. Methodology & Standards
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

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
_TEMPLATE_ID = "gri_305"
_PRIMARY = "#00695c"
_SECONDARY = "#00897b"
_ACCENT = "#26a69a"
_LIGHT = "#e0f2f1"
_LIGHTER = "#f0faf9"

GRI_305_DISCLOSURES = [
    {"code": "305-1", "title": "Direct (Scope 1) GHG emissions", "type": "Quantitative"},
    {"code": "305-2", "title": "Energy indirect (Scope 2) GHG emissions", "type": "Quantitative"},
    {"code": "305-3", "title": "Other indirect (Scope 3) GHG emissions", "type": "Quantitative"},
    {"code": "305-4", "title": "GHG emissions intensity", "type": "Quantitative"},
    {"code": "305-5", "title": "Reduction of GHG emissions", "type": "Quantitative"},
    {"code": "305-6", "title": "Emissions of ozone-depleting substances (ODS)", "type": "Quantitative"},
    {"code": "305-7", "title": "Nitrogen oxides (NOx), sulfur oxides (SOx), and other", "type": "Quantitative"},
]

XBRL_TAGS: Dict[str, str] = {
    "scope1": "gl:GRI305_1_Scope1", "scope2_location": "gl:GRI305_2_Scope2Location",
    "scope2_market": "gl:GRI305_2_Scope2Market", "scope3": "gl:GRI305_3_Scope3",
    "intensity": "gl:GRI305_4_Intensity", "reduction": "gl:GRI305_5_Reduction",
}

def _utcnow(): return datetime.now(timezone.utc).replace(microsecond=0)
def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(data):
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def _dec(val, places=2):
    try:
        d = Decimal(str(val)); q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except: return str(val)
def _dec_comma(val, places=2):
    try:
        d = Decimal(str(val)); q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP); parts = str(rounded).split(".")
        int_part = parts[0]; negative = int_part.startswith("-")
        if negative: int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0: formatted = "," + formatted
            formatted = ch + formatted
        if negative: formatted = "-" + formatted
        if len(parts) > 1: formatted += "." + parts[1]
        return formatted
    except: return str(val)


class GRI305Template:
    """GRI 305 Emissions Disclosure template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_305_1(data), self._md_305_2(data), self._md_305_3(data),
            self._md_305_4(data), self._md_305_5(data), self._md_305_6(data),
            self._md_305_7(data), self._md_content_index(data),
            self._md_methodology(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [self._html_header(data), self._html_executive_summary(data),
                 self._html_305_1(data), self._html_305_2(data), self._html_305_3(data),
                 self._html_305_5(data), self._html_content_index(data),
                 self._html_xbrl(data), self._html_audit(data), self._html_footer(data)]
        body = "\n".join(parts)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>GRI 305 - {data.get("org_name","")}</title>\n<style>\n{css}\n</style>\n</head>\n'
                f'<body>\n<div class="report">\n{body}\n</div>\n<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "reporting_year": data.get("reporting_year",""),
            "framework": "GRI", "standard": "GRI 305",
            "disclosures": {
                "305_1": data.get("scope1_detail", {}), "305_2": data.get("scope2_detail", {}),
                "305_3": data.get("scope3_detail", {}), "305_4": data.get("intensity", {}),
                "305_5": data.get("reductions", {}), "305_6": data.get("ods", {}),
                "305_7": data.get("air_emissions", {}),
            },
            "content_index": [{"code": d["code"], "title": d["title"]} for d in GRI_305_DISCLOSURES],
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"GRI 305 - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# GRI 305 Emissions Disclosure\n\n**Organization:** {data.get('org_name','')}  \n**Reporting Year:** {data.get('reporting_year','')}  \n**Standard:** GRI 305: Emissions 2016  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        s1 = float(data.get("scope1",0)); s2l = float(data.get("scope2_location",0)); s3 = float(data.get("scope3",0))
        total = s1+s2l+s3; red = data.get("reductions",{}).get("total",0)
        lines = ["## 1. Executive Summary\n", "| Metric | Value |", "|--------|-------|",
                  f"| Scope 1 (305-1) | {_dec_comma(s1,0)} tCO2e |", f"| Scope 2 (305-2, Location) | {_dec_comma(s2l,0)} tCO2e |",
                  f"| Scope 3 (305-3) | {_dec_comma(s3,0)} tCO2e |", f"| Total | {_dec_comma(total,0)} tCO2e |",
                  f"| Reductions (305-5) | {_dec_comma(red,0)} tCO2e |"]
        return "\n".join(lines)

    def _md_305_1(self, data):
        detail = data.get("scope1_detail", {})
        s1 = float(data.get("scope1",0))
        by_source = detail.get("by_source", {})
        by_ghg = detail.get("by_ghg", {})
        lines = ["## 2. GRI 305-1: Direct (Scope 1) GHG Emissions\n",
                  f"**Gross direct (Scope 1) emissions:** {_dec_comma(s1,0)} tCO2e\n",
                  "### By Source\n", "| Source | Emissions (tCO2e) |", "|--------|------------------:|"]
        for src, em in by_source.items():
            lines.append(f"| {src} | {_dec_comma(em,0)} |")
        if by_ghg:
            lines.extend(["\n### By GHG\n", "| Gas | Emissions (tCO2e) |", "|-----|------------------:|"])
            for gas, em in by_ghg.items():
                lines.append(f"| {gas} | {_dec_comma(em,0)} |")
        lines.extend(["\n### Methodology\n", "| Parameter | Value |", "|-----------|-------|",
                       f"| Standard | {detail.get('standard','GHG Protocol')} |",
                       f"| Consolidation | {detail.get('consolidation','Operational control')} |",
                       f"| EF Source | {detail.get('ef_source','IPCC AR6, DEFRA')} |",
                       f"| GWP | {detail.get('gwp','IPCC AR6 100-year')} |"])
        return "\n".join(lines)

    def _md_305_2(self, data):
        s2l = float(data.get("scope2_location",0)); s2m = float(data.get("scope2_market",0))
        detail = data.get("scope2_detail", {})
        lines = ["## 3. GRI 305-2: Energy Indirect (Scope 2) GHG Emissions\n",
                  f"**Location-based:** {_dec_comma(s2l,0)} tCO2e  \n**Market-based:** {_dec_comma(s2m,0)} tCO2e\n",
                  "### By Energy Type\n", "| Type | Location (tCO2e) | Market (tCO2e) |", "|------|:----------------:|:--------------:|"]
        for etype, vals in detail.get("by_type", {}).items():
            lines.append(f"| {etype} | {_dec_comma(vals.get('location',0),0)} | {_dec_comma(vals.get('market',0),0)} |")
        return "\n".join(lines)

    def _md_305_3(self, data):
        s3 = float(data.get("scope3",0))
        detail = data.get("scope3_detail", {})
        by_cat = detail.get("by_category", {})
        lines = ["## 4. GRI 305-3: Other Indirect (Scope 3) GHG Emissions\n",
                  f"**Total Scope 3:** {_dec_comma(s3,0)} tCO2e\n",
                  "| Category | Name | Emissions (tCO2e) | Method |",
                  "|:--------:|------|------------------:|--------|"]
        for cat_num, info in sorted(by_cat.items(), key=lambda x: int(x[0])):
            lines.append(f"| {cat_num} | {info.get('name','')} | {_dec_comma(info.get('emissions',0),0)} | {info.get('method','Spend-based')} |")
        return "\n".join(lines)

    def _md_305_4(self, data):
        intensity = data.get("intensity", {})
        lines = ["## 5. GRI 305-4: GHG Emissions Intensity\n",
                  "| Metric | Value | Denominator |", "|--------|-------|-------------|"]
        for name, info in intensity.items():
            lines.append(f"| {name.replace('_',' ').title()} | {_dec(info.get('value',0))} | {info.get('denominator','')} |")
        if not intensity: lines.append("| _No intensity metrics_ | - | - |")
        return "\n".join(lines)

    def _md_305_5(self, data):
        reductions = data.get("reductions", {})
        initiatives = reductions.get("initiatives", [])
        lines = ["## 6. GRI 305-5: Reduction of GHG Emissions\n",
                  f"**Total Reduction:** {_dec_comma(reductions.get('total',0),0)} tCO2e  \n"
                  f"**Reduction from Baseline:** {_dec(reductions.get('pct_from_baseline',0))}%\n",
                  "| # | Initiative | Reduction (tCO2e) | Scope | Type |",
                  "|---|-----------|------------------:|-------|------|"]
        for i, init in enumerate(initiatives, 1):
            lines.append(f"| {i} | {init.get('name','')} | {_dec_comma(init.get('reduction',0),0)} | {init.get('scope','')} | {init.get('type','Direct')} |")
        if not initiatives: lines.append("| - | _No initiatives_ | - | - | - |")
        return "\n".join(lines)

    def _md_305_6(self, data):
        ods = data.get("ods", {})
        lines = ["## 7. GRI 305-6: Emissions of Ozone-Depleting Substances\n",
                  "| Parameter | Value |", "|-----------|-------|",
                  f"| Production | {_dec(ods.get('production',0))} tonnes CFC-11 eq |",
                  f"| Imports | {_dec(ods.get('imports',0))} tonnes CFC-11 eq |",
                  f"| Exports | {_dec(ods.get('exports',0))} tonnes CFC-11 eq |",
                  f"| Total | {_dec(ods.get('total',0))} tonnes CFC-11 eq |"]
        return "\n".join(lines)

    def _md_305_7(self, data):
        air = data.get("air_emissions", {})
        lines = ["## 8. GRI 305-7: NOx, SOx, and Other Air Emissions\n",
                  "| Pollutant | Emissions (tonnes) |", "|-----------|-------------------:|",
                  f"| NOx | {_dec(air.get('nox',0))} |", f"| SOx | {_dec(air.get('sox',0))} |",
                  f"| PM | {_dec(air.get('pm',0))} |", f"| VOC | {_dec(air.get('voc',0))} |",
                  f"| HAP | {_dec(air.get('hap',0))} |"]
        return "\n".join(lines)

    def _md_content_index(self, data):
        lines = ["## 9. GRI Content Index\n",
                  "| Code | Disclosure | Page/Location | Omissions |",
                  "|------|-----------|---------------|-----------|"]
        for d in GRI_305_DISCLOSURES:
            lines.append(f"| {d['code']} | {d['title']} | Section {GRI_305_DISCLOSURES.index(d)+2} | None |")
        return "\n".join(lines)

    def _md_methodology(self, data):
        meth = data.get("methodology", {})
        lines = ["## 10. Methodology & Standards\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Reporting Standard | {meth.get('standard','GRI 305: Emissions 2016')} |",
                  f"| GHG Protocol | {meth.get('ghg_protocol','Corporate Standard + Scope 3')} |",
                  f"| Consolidation | {meth.get('consolidation','Operational control')} |",
                  f"| EF Sources | {meth.get('ef_sources','IPCC AR6, IEA, DEFRA')} |",
                  f"| GWP | {meth.get('gwp','IPCC AR6 100-year')} |",
                  f"| Assurance | {meth.get('assurance','Limited assurance (ISAE 3410)')} |"]
        return "\n".join(lines)

    def _md_xbrl(self, data):
        lines = ["## 11. XBRL Tagging Summary\n", "| Data Point | XBRL Tag |", "|------------|----------|"]
        for key, tag in XBRL_TAGS.items(): lines.append(f"| {key.replace('_',' ').title()} | {tag} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f"## 12. Audit Trail & Provenance\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*GRI 305 emissions disclosure.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #b2dfdb;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#e8f5e9;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>GRI 305 Emissions Disclosure</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2l=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        return (f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">305-1</div><div class="card-value">{_dec_comma(s1,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">305-2</div><div class="card-value">{_dec_comma(s2l,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">305-3</div><div class="card-value">{_dec_comma(s3,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(s1+s2l+s3,0)}</div><div class="card-unit">tCO2e</div></div>\n</div>')

    def _html_305_1(self, data):
        s1 = float(data.get("scope1",0)); by_source = data.get("scope1_detail",{}).get("by_source",{})
        rows = ""
        for src, em in by_source.items(): rows += f'<tr><td>{src}</td><td>{_dec_comma(em,0)}</td></tr>\n'
        return f'<h2>2. 305-1 Scope 1</h2>\n<p>Total: {_dec_comma(s1,0)} tCO2e</p>\n<table>\n<tr><th>Source</th><th>Emissions</th></tr>\n{rows}</table>'

    def _html_305_2(self, data):
        s2l=float(data.get("scope2_location",0)); s2m=float(data.get("scope2_market",0))
        return (f'<h2>3. 305-2 Scope 2</h2>\n<table>\n<tr><th>Method</th><th>Emissions</th></tr>\n'
                f'<tr><td>Location</td><td>{_dec_comma(s2l,0)}</td></tr>\n<tr><td>Market</td><td>{_dec_comma(s2m,0)}</td></tr>\n</table>')

    def _html_305_3(self, data):
        by_cat = data.get("scope3_detail",{}).get("by_category",{})
        rows = ""
        for cat, info in sorted(by_cat.items(), key=lambda x: int(x[0])):
            rows += f'<tr><td>{cat}</td><td>{info.get("name","")}</td><td>{_dec_comma(info.get("emissions",0),0)}</td></tr>\n'
        return f'<h2>4. 305-3 Scope 3</h2>\n<table>\n<tr><th>Cat</th><th>Name</th><th>Emissions</th></tr>\n{rows}</table>'

    def _html_305_5(self, data):
        reductions = data.get("reductions",{})
        return f'<h2>5. 305-5 Reductions</h2>\n<p>Total: {_dec_comma(reductions.get("total",0),0)} tCO2e ({_dec(reductions.get("pct_from_baseline",0))}% from baseline)</p>'

    def _html_content_index(self, data):
        rows = ""
        for d in GRI_305_DISCLOSURES: rows += f'<tr><td>{d["code"]}</td><td>{d["title"]}</td><td>Disclosed</td></tr>\n'
        return f'<h2>6. Content Index</h2>\n<table>\n<tr><th>Code</th><th>Disclosure</th><th>Status</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data):
        rows = ""
        for key, tag in XBRL_TAGS.items(): rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>7. XBRL</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>8. Audit</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n<tr><td>ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - GRI 305</div>'
