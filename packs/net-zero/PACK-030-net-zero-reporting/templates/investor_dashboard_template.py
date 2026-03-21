# -*- coding: utf-8 -*-
"""
InvestorDashboardTemplate - Investor Dashboard Template for PACK-030.

Renders an investor-focused climate dashboard with TCFD + ISSB alignment,
financial materiality assessment, scenario analysis summary, ESG ratings
integration, key climate metrics, and portfolio-level carbon analytics.
Multi-format output (MD, HTML, JSON, PDF) with SHA-256 provenance.

Sections:
    1.  Executive Summary (Investor KPIs)
    2.  TCFD Alignment Status
    3.  Emissions Performance & Trend
    4.  Targets & Progress
    5.  Scenario Analysis Summary
    6.  Financial Materiality Assessment
    7.  ESG Ratings & Benchmarks
    8.  Climate Risk Heatmap
    9.  Audit Trail & Provenance

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
_TEMPLATE_ID = "investor_dashboard"
_PRIMARY = "#0d3b66"
_SECONDARY = "#1a6b8a"
_ACCENT = "#f4a261"
_LIGHT = "#e3f0f7"
_LIGHTER = "#f4f9fc"

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
def _pct_change(current, baseline):
    c, b = Decimal(str(current)), Decimal(str(baseline))
    if b == 0: return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

TCFD_PILLARS = ["Governance", "Strategy", "Risk Management", "Metrics & Targets"]

class InvestorDashboardTemplate:
    """Investor dashboard template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_tcfd_status(data), self._md_emissions(data),
            self._md_targets(data), self._md_scenarios(data),
            self._md_materiality(data), self._md_esg_ratings(data),
            self._md_risk_heatmap(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [self._html_header(data), self._html_executive_summary(data),
                 self._html_tcfd_status(data), self._html_emissions(data),
                 self._html_targets(data), self._html_audit(data), self._html_footer(data)]
        body = "\n".join(parts)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>Investor Dashboard - {data.get("org_name","")}</title>\n<style>\n{css}\n</style>\n</head>\n'
                f'<body>\n<div class="report">\n{body}\n</div>\n<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name",""), "reporting_year": data.get("reporting_year",""),
            "stakeholder": "investor",
            "emissions": data.get("emissions", {}), "targets": data.get("targets", []),
            "tcfd_alignment": data.get("tcfd_alignment", {}),
            "scenarios": data.get("scenarios", {}), "esg_ratings": data.get("esg_ratings", {}),
            "materiality": data.get("materiality", {}),
        }
        result["provenance_hash"] = _compute_hash(result); return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data), "structured_data": self.render_json(data),
                "metadata": {"title": f"Investor Dashboard - {data.get('org_name','')}", "author": "GreenLang PACK-030"}}

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Investor Climate Dashboard\n\n**Organization:** {data.get('org_name','')}  \n**Reporting Year:** {data.get('reporting_year','')}  \n**Audience:** Institutional Investors  \n**Report Date:** {ts}  \n**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"

    def _md_executive_summary(self, data):
        em = data.get("emissions", {}); s1=float(em.get("scope1",0)); s2=float(em.get("scope2",0)); s3=float(em.get("scope3",0))
        total = s1+s2+s3; prev = float(em.get("previous_total", total))
        yoy = float(_pct_change(total, prev)) if prev else 0
        lines = ["## 1. Investor KPIs\n", "| KPI | Value |", "|-----|-------|",
                  f"| Total Emissions | {_dec_comma(total,0)} tCO2e |",
                  f"| YoY Change | {'+' if yoy > 0 else ''}{_dec(yoy)}% |",
                  f"| Carbon Intensity | {em.get('intensity','N/A')} |",
                  f"| Net-Zero Target | {data.get('net_zero_year','')} |",
                  f"| SBTi Status | {data.get('sbti_status','Validated')} |",
                  f"| TCFD Alignment | {data.get('tcfd_alignment',{}).get('overall','Aligned')} |"]
        return "\n".join(lines)

    def _md_tcfd_status(self, data):
        alignment = data.get("tcfd_alignment", {})
        lines = ["## 2. TCFD Alignment Status\n", "| Pillar | Status | Score |", "|--------|--------|:-----:|"]
        for pillar in TCFD_PILLARS:
            key = pillar.lower().replace(" & ", "_").replace(" ", "_")
            lines.append(f"| {pillar} | {alignment.get(key, {}).get('status','Aligned')} | {alignment.get(key, {}).get('score','N/A')} |")
        lines.append(f"\n**Overall:** {alignment.get('overall','Aligned')}")
        return "\n".join(lines)

    def _md_emissions(self, data):
        em = data.get("emissions", {}); history = data.get("emissions_history", [])
        lines = ["## 3. Emissions Performance & Trend\n",
                  "| Year | Total (tCO2e) | YoY (%) |", "|------|:-------------:|--------:|"]
        prev = None
        for h in history:
            t = float(h.get("total",0))
            if prev: yoy = float(_pct_change(t, prev)); lines.append(f"| {h.get('year','')} | {_dec_comma(t,0)} | {'+' if yoy > 0 else ''}{_dec(yoy)}% |")
            else: lines.append(f"| {h.get('year','')} | {_dec_comma(t,0)} | - |")
            prev = t
        return "\n".join(lines)

    def _md_targets(self, data):
        targets = data.get("targets", [])
        lines = ["## 4. Targets & Progress\n",
                  "| # | Target | Year | Progress (%) | On Track |",
                  "|---|--------|:----:|:------------:|:--------:|"]
        for i, t in enumerate(targets, 1):
            lines.append(f"| {i} | {t.get('name','')} | {t.get('year','')} | {_dec(t.get('progress',0))}% | {t.get('on_track','Yes')} |")
        return "\n".join(lines)

    def _md_scenarios(self, data):
        scenarios = data.get("scenarios", {})
        lines = ["## 5. Scenario Analysis Summary\n",
                  "| Scenario | Revenue Impact | Asset Risk | Overall |", "|----------|---------------|-----------|---------|"]
        for name, s in scenarios.items():
            lines.append(f"| {name} | {s.get('revenue_impact','')} | {s.get('asset_risk','')} | {s.get('overall','')} |")
        return "\n".join(lines)

    def _md_materiality(self, data):
        mat = data.get("materiality", {})
        items = mat.get("items", [])
        lines = ["## 6. Financial Materiality Assessment\n",
                  "| # | Issue | Impact | Likelihood | Financial Exposure |",
                  "|---|-------|--------|------------|-------------------|"]
        for i, m in enumerate(items, 1):
            lines.append(f"| {i} | {m.get('issue','')} | {m.get('impact','')} | {m.get('likelihood','')} | {m.get('exposure','')} |")
        return "\n".join(lines)

    def _md_esg_ratings(self, data):
        ratings = data.get("esg_ratings", {})
        lines = ["## 7. ESG Ratings & Benchmarks\n", "| Provider | Rating | Peer Rank |", "|----------|--------|-----------|"]
        for provider, info in ratings.items():
            lines.append(f"| {provider} | {info.get('rating','')} | {info.get('peer_rank','')} |")
        return "\n".join(lines)

    def _md_risk_heatmap(self, data):
        risks = data.get("risk_heatmap", [])
        lines = ["## 8. Climate Risk Heatmap\n",
                  "| Risk | Likelihood | Impact | Rating |", "|------|------------|--------|:------:|"]
        for r in risks:
            lines.append(f"| {r.get('name','')} | {r.get('likelihood','')} | {r.get('impact','')} | {r.get('rating','')} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f"## 9. Audit Trail\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*Investor climate dashboard.*"

    def _css(self):
        return (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
                f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
                f"h1{{color:{_PRIMARY};border-bottom:3px solid {_ACCENT};padding-bottom:12px;}}"
                f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
                f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
                f"th,td{{border:1px solid #b3d4e6;padding:10px 14px;text-align:left;}}"
                f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
                f"tr:nth-child(even){{background:#f0f7fb;}}"
                f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
                f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_ACCENT};}}"
                f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
                f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
                f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_SECONDARY};font-size:0.85em;text-align:center;}}")

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Investor Climate Dashboard</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data):
        em = data.get("emissions",{}); s1=float(em.get("scope1",0)); s2=float(em.get("scope2",0)); s3=float(em.get("scope3",0))
        return (f'<h2>1. KPIs</h2>\n<div class="summary-cards">\n'
                f'<div class="card"><div class="card-label">Total Emissions</div><div class="card-value">{_dec_comma(s1+s2+s3,0)}</div></div>\n'
                f'<div class="card"><div class="card-label">SBTi</div><div class="card-value">{data.get("sbti_status","Validated")}</div></div>\n'
                f'<div class="card"><div class="card-label">Net-Zero</div><div class="card-value">{data.get("net_zero_year","")}</div></div>\n</div>')

    def _html_tcfd_status(self, data):
        alignment = data.get("tcfd_alignment",{}); rows = ""
        for pillar in TCFD_PILLARS:
            key = pillar.lower().replace(" & ","_").replace(" ","_")
            rows += f'<tr><td>{pillar}</td><td>{alignment.get(key,{}).get("status","Aligned")}</td></tr>\n'
        return f'<h2>2. TCFD Status</h2>\n<table>\n<tr><th>Pillar</th><th>Status</th></tr>\n{rows}</table>'

    def _html_emissions(self, data):
        history = data.get("emissions_history",[]); rows = ""
        for h in history: rows += f'<tr><td>{h.get("year","")}</td><td>{_dec_comma(h.get("total",0),0)}</td></tr>\n'
        return f'<h2>3. Emissions Trend</h2>\n<table>\n<tr><th>Year</th><th>Total</th></tr>\n{rows}</table>'

    def _html_targets(self, data):
        targets = data.get("targets",[]); rows = ""
        for i, t in enumerate(targets, 1):
            rows += f'<tr><td>{i}</td><td>{t.get("name","")}</td><td>{_dec(t.get("progress",0))}%</td><td>{t.get("on_track","")}</td></tr>\n'
        return f'<h2>4. Targets</h2>\n<table>\n<tr><th>#</th><th>Target</th><th>Progress</th><th>On Track</th></tr>\n{rows}</table>'

    def _html_audit(self, data):
        rid = _new_uuid(); ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""; dh = _compute_hash(data)
        return f'<h2>5. Audit</h2>\n<table>\n<tr><th>Param</th><th>Value</th></tr>\n<tr><td>ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - Investor Dashboard</div>'
