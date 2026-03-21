# -*- coding: utf-8 -*-
"""
TCFDMetricsTemplate - TCFD Metrics & Targets Pillar Template for PACK-030.

Renders TCFD Metrics & Targets pillar disclosure covering Scope 1/2/3
emissions, emissions intensity, climate-related targets, progress
tracking, key climate metrics, internal carbon pricing, and remuneration
linkage. Multi-format output (MD, HTML, JSON, PDF) with SHA-256
provenance hashing.

Sections:
    1.  Executive Summary
    2.  GHG Emissions by Scope
    3.  Emissions Intensity Metrics
    4.  Climate-Related Targets
    5.  Target Progress Tracking
    6.  Key Climate Metrics
    7.  Internal Carbon Pricing
    8.  Remuneration Linkage
    9.  Year-over-Year Trend
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
_TEMPLATE_ID = "tcfd_metrics"
_PRIMARY = "#1a237e"
_SECONDARY = "#283593"
_ACCENT = "#42a5f5"
_LIGHT = "#e8eaf6"
_LIGHTER = "#f5f5ff"

XBRL_TAGS: Dict[str, str] = {
    "scope1": "gl:TCFDMetricsScope1", "scope2_location": "gl:TCFDMetricsScope2Location",
    "scope2_market": "gl:TCFDMetricsScope2Market", "scope3": "gl:TCFDMetricsScope3",
    "intensity_revenue": "gl:TCFDMetricsIntensityRevenue",
    "internal_carbon_price": "gl:TCFDMetricsInternalCarbonPrice",
    "targets_count": "gl:TCFDMetricsTargetsCount",
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
def _pct_change(current, baseline):
    c, b = Decimal(str(current)), Decimal(str(baseline))
    if b == 0: return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class TCFDMetricsTemplate:
    """TCFD Metrics & Targets template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_emissions(data), self._md_intensity(data),
            self._md_targets(data), self._md_progress(data),
            self._md_key_metrics(data), self._md_carbon_pricing(data),
            self._md_remuneration(data), self._md_trend(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_emissions(data), self._html_targets(data),
            self._html_trend(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>TCFD Metrics - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        s1 = float(data.get("scope1", 0)); s2l = float(data.get("scope2_location", 0))
        s2m = float(data.get("scope2_market", 0)); s3 = float(data.get("scope3", 0))
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "reporting_year": data.get("reporting_year",""),
            "framework": "TCFD", "pillar": "Metrics & Targets",
            "emissions": {"scope1": str(s1), "scope2_location": str(s2l), "scope2_market": str(s2m), "scope3": str(s3), "total_location": str(s1+s2l+s3)},
            "intensity": data.get("intensity_metrics", {}), "targets": data.get("targets", []),
            "carbon_pricing": data.get("carbon_pricing", {}),
            "key_metrics": data.get("key_metrics", []), "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"TCFD Metrics - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# TCFD Metrics & Targets Disclosure\n\n**Organization:** {data.get('org_name','')}  \n**Reporting Year:** {data.get('reporting_year','')}  \n**Framework:** TCFD | **Pillar:** Metrics & Targets  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        s1 = float(data.get("scope1", 0)); s2l = float(data.get("scope2_location", 0)); s3 = float(data.get("scope3", 0))
        total = s1 + s2l + s3
        lines = ["## 1. Executive Summary\n", "| Metric | Value |", "|--------|-------|",
                  f"| Scope 1 | {_dec_comma(s1,0)} tCO2e |", f"| Scope 2 (Location) | {_dec_comma(s2l,0)} tCO2e |",
                  f"| Scope 3 | {_dec_comma(s3,0)} tCO2e |", f"| Total | {_dec_comma(total,0)} tCO2e |",
                  f"| Targets Set | {len(data.get('targets',[]))} |",
                  f"| Internal Carbon Price | {data.get('carbon_pricing',{}).get('price','N/A')} |"]
        return "\n".join(lines)

    def _md_emissions(self, data):
        s1 = float(data.get("scope1",0)); s2l = float(data.get("scope2_location",0))
        s2m = float(data.get("scope2_market",0)); s3 = float(data.get("scope3",0))
        total_l = s1+s2l+s3; total_m = s1+s2m+s3
        lines = ["## 2. GHG Emissions by Scope\n",
                  "| Scope | Method | Emissions (tCO2e) | Share (%) |", "|-------|--------|------------------:|----------:|",
                  f"| Scope 1 | Direct | {_dec_comma(s1,0)} | {_dec(s1/total_l*100 if total_l else 0)}% |",
                  f"| Scope 2 | Location | {_dec_comma(s2l,0)} | {_dec(s2l/total_l*100 if total_l else 0)}% |",
                  f"| Scope 2 | Market | {_dec_comma(s2m,0)} | - |",
                  f"| Scope 3 | Indirect | {_dec_comma(s3,0)} | {_dec(s3/total_l*100 if total_l else 0)}% |",
                  f"| **Total (Location)** | | **{_dec_comma(total_l,0)}** | **100%** |",
                  f"| **Total (Market)** | | **{_dec_comma(total_m,0)}** | |"]
        return "\n".join(lines)

    def _md_intensity(self, data):
        metrics = data.get("intensity_metrics", {})
        lines = ["## 3. Emissions Intensity Metrics\n", "| Metric | Value | Unit |", "|--------|-------|------|"]
        for name, info in metrics.items():
            lines.append(f"| {name.replace('_',' ').title()} | {_dec(info.get('value',0))} | {info.get('unit','')} |")
        if not metrics:
            lines.append("| _No intensity metrics provided_ | - | - |")
        return "\n".join(lines)

    def _md_targets(self, data):
        targets = data.get("targets", [])
        lines = ["## 4. Climate-Related Targets\n",
                  "| # | Target | Scope | Base Year | Target Year | Reduction (%) | SBTi |",
                  "|---|--------|-------|:---------:|:-----------:|--------------:|:----:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {t.get('scope','')} | {t.get('base_year','')} | {t.get('target_year','')} | {_dec(t.get('reduction_pct',0))}% | {t.get('sbti','No')} |")
        if not targets: lines.append("| - | _No targets set_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_progress(self, data):
        targets = data.get("targets", [])
        lines = ["## 5. Target Progress Tracking\n",
                  "| # | Target | Progress (%) | On Track | Gap (tCO2e) |",
                  "|---|--------|-------------:|:--------:|------------:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {_dec(t.get('progress_pct',0))}% | {t.get('on_track','N/A')} | {_dec_comma(t.get('gap_tco2e',0),0)} |")
        return "\n".join(lines)

    def _md_key_metrics(self, data):
        metrics = data.get("key_metrics", [])
        lines = ["## 6. Key Climate Metrics\n", "| # | Metric | Value | Unit | Trend |", "|---|--------|-------|------|-------|"]
        for i, m in enumerate(metrics, 1):
            lines.append(f"| {i} | {m.get('name','')} | {_dec(m.get('value',0))} | {m.get('unit','')} | {m.get('trend','')} |")
        if not metrics: lines.append("| - | _No key metrics_ | - | - | - |")
        return "\n".join(lines)

    def _md_carbon_pricing(self, data):
        cp = data.get("carbon_pricing", {})
        lines = ["## 7. Internal Carbon Pricing\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Carbon Price Used | {cp.get('used','Yes')} |", f"| Price | {cp.get('price','$50/tCO2e')} |",
                  f"| Type | {cp.get('type','Shadow price')} |", f"| Application | {cp.get('application','Capital investment decisions')} |",
                  f"| Review Frequency | {cp.get('review','Annual')} |"]
        return "\n".join(lines)

    def _md_remuneration(self, data):
        remuneration = data.get("remuneration", {})
        lines = ["## 8. Remuneration Linkage\n", "| Parameter | Value |", "|-----------|-------|",
                  f"| Linked to Climate | {remuneration.get('linked','Yes')} |",
                  f"| Who | {remuneration.get('who','CEO, CSO, COO')} |",
                  f"| Metric | {remuneration.get('metric','Scope 1+2 reduction target')} |",
                  f"| Weight | {remuneration.get('weight','15% of bonus')} |"]
        return "\n".join(lines)

    def _md_trend(self, data):
        history = data.get("historical_emissions", [])
        lines = ["## 9. Year-over-Year Trend\n", "| Year | Total (tCO2e) | YoY Change (%) |", "|------|:-------------:|:---------------:|"]
        prev = None
        for h in history:
            em = float(h.get("total", 0))
            if prev: pct = float(_pct_change(em, prev)); lines.append(f"| {h.get('year','')} | {_dec_comma(em,0)} | {'+' if pct > 0 else ''}{_dec(pct)}% |")
            else: lines.append(f"| {h.get('year','')} | {_dec_comma(em,0)} | - |")
            prev = em
        if not history: lines.append("| - | _No history_ | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data):
        lines = ["## 10. XBRL Tagging Summary\n", "| Data Point | XBRL Tag |", "|------------|----------|"]
        for key, tag in XBRL_TAGS.items(): lines.append(f"| {key.replace('_',' ').title()} | {tag} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f"## 11. Audit Trail & Provenance\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*TCFD Metrics & Targets pillar disclosure.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #c5cae9;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#f3f4fb;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>TCFD Metrics & Targets</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        s1=float(data.get("scope1",0)); s2l=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        return (f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(s1,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(s2l,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Scope 3</div><div class="card-value">{_dec_comma(s3,0)}</div><div class="card-unit">tCO2e</div></div>\n'
                f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(s1+s2l+s3,0)}</div><div class="card-unit">tCO2e</div></div>\n</div>')

    def _html_emissions(self, data):
        s1=float(data.get("scope1",0)); s2l=float(data.get("scope2_location",0)); s3=float(data.get("scope3",0))
        return (f'<h2>2. Emissions</h2>\n<table>\n<tr><th>Scope</th><th>Emissions</th></tr>\n'
                f'<tr><td>Scope 1</td><td>{_dec_comma(s1,0)}</td></tr>\n<tr><td>Scope 2 (Loc)</td><td>{_dec_comma(s2l,0)}</td></tr>\n'
                f'<tr><td>Scope 3</td><td>{_dec_comma(s3,0)}</td></tr>\n<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2l+s3,0)}</strong></td></tr>\n</table>')

    def _html_targets(self, data):
        targets = data.get("targets", []); rows = ""
        for i, t in enumerate(targets, 1):
            rows += f'<tr><td>{i}</td><td>{t.get("name","")}</td><td>{t.get("scope","")}</td><td>{_dec(t.get("reduction_pct",0))}%</td><td>{_dec(t.get("progress_pct",0))}%</td></tr>\n'
        return f'<h2>3. Targets</h2>\n<table>\n<tr><th>#</th><th>Target</th><th>Scope</th><th>Reduction</th><th>Progress</th></tr>\n{rows}</table>'

    def _html_trend(self, data):
        history = data.get("historical_emissions", []); rows = ""; prev = None
        for h in history:
            em = float(h.get("total", 0))
            if prev: pct = float(_pct_change(em, prev)); rows += f'<tr><td>{h.get("year","")}</td><td>{_dec_comma(em,0)}</td><td>{_dec(pct)}%</td></tr>\n'
            else: rows += f'<tr><td>{h.get("year","")}</td><td>{_dec_comma(em,0)}</td><td>-</td></tr>\n'
            prev = em
        return f'<h2>4. Trend</h2>\n<table>\n<tr><th>Year</th><th>Total</th><th>YoY %</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data):
        rows = ""
        for key, tag in XBRL_TAGS.items(): rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>5. XBRL</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>6. Audit</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - TCFD Metrics</div>'
