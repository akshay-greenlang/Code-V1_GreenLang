# -*- coding: utf-8 -*-
"""
ExecutiveDashboardTemplate - Board-level climate dashboard for PACK-027.

Renders a single-page executive climate dashboard with 15-20 key KPIs,
traffic-light status indicators, trend sparklines, peer benchmarking,
and strategic commentary. Designed for quarterly board reporting.

Sections:
    1. Headline KPIs (total emissions, intensity, target progress)
    2. Scope Mix (pie-chart-ready data)
    3. Target Pathway Progress (actual vs. SBTi pathway)
    4. Traffic Light Indicators (Red/Amber/Green for key areas)
    5. Initiative Status (energy, fleet, procurement, supply chain)
    6. Regulatory Compliance Dashboard
    7. Peer Benchmarking
    8. Key Risks & Opportunities
    9. Next Quarter Priorities

Output: Markdown, HTML, JSON
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "executive_dashboard"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"
_RED = "#c62828"
_AMBER = "#ef6c00"
_GREEN = "#2e7d32"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return str(round(float(val), 1)) + "%"
    except Exception:
        return str(val)

def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default

def _rag_status(status: str) -> str:
    mapping = {"green": "GREEN", "amber": "AMBER", "red": "RED"}
    return mapping.get(status.lower(), status.upper())

def _rag_md(status: str) -> str:
    s = status.lower()
    if s == "green":
        return "**[GREEN]**"
    elif s == "amber":
        return "**[AMBER]**"
    elif s == "red":
        return "**[RED]**"
    return f"**[{status.upper()}]**"

class ExecutiveDashboardTemplate:
    """
    Board-level enterprise climate dashboard template.

    Single-page executive summary with 15-20 KPIs, traffic-light RAG
    indicators, initiative tracking, regulatory compliance, and peer
    benchmarking. Supports Markdown, HTML, and JSON output.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_headline_kpis(data),
            self._md_target_progress(data),
            self._md_rag_indicators(data),
            self._md_initiative_status(data),
            self._md_regulatory_compliance(data),
            self._md_peer_benchmark(data),
            self._md_risks_opportunities(data),
            self._md_next_quarter(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0;}}"
            f".kpi{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".kpi-label{{font-size:0.75em;color:#37474f;text-transform:uppercase;}}"
            f".kpi-value{{font-size:1.6em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".kpi-unit{{font-size:0.7em;color:#607d8b;}}"
            f".rag-green{{color:{_GREEN};font-weight:700;}}"
            f".rag-amber{{color:{_AMBER};font-weight:700;}}"
            f".rag-red{{color:{_RED};font-weight:700;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        target_progress = float(data.get("target_progress_pct", 0))

        body = (
            f'<h1>Executive Climate Dashboard</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'Q{data.get("quarter", "?")} {data.get("reporting_year", "")} | '
            f'Generated: {self.generated_at.strftime("%Y-%m-%d %H:%M UTC")}</p>\n'
            f'<div class="kpi-grid">\n'
            f'  <div class="kpi"><div class="kpi-label">Total Emissions</div>'
            f'<div class="kpi-value">{_dec_comma(total)}</div><div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 1</div>'
            f'<div class="kpi-value">{_dec_comma(s1)}</div><div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 2</div>'
            f'<div class="kpi-value">{_dec_comma(s2)}</div><div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 3</div>'
            f'<div class="kpi-value">{_dec_comma(s3)}</div><div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Target Progress</div>'
            f'<div class="kpi-value">{_pct(target_progress)}</div><div class="kpi-unit">of pathway</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Data Quality</div>'
            f'<div class="kpi-value">{data.get("overall_dq_score", 0)}</div><div class="kpi-unit">/100</div></div>\n'
            f'</div>\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Executive Dashboard | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Executive Climate Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": data.get("org_name", ""),
            "quarter": data.get("quarter", ""),
            "reporting_year": data.get("reporting_year", ""),
            "kpis": {
                "total_tco2e": round(total, 2),
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "intensity_per_employee": round(_safe_div(total, employees), 2),
                "intensity_per_revenue": round(
                    _safe_div(total, float(data.get("revenue", 1))) * 1_000_000, 2
                ),
                "target_progress_pct": data.get("target_progress_pct", 0),
                "yoy_change_pct": data.get("yoy_change_pct", 0),
                "renewable_energy_pct": data.get("renewable_energy_pct", 0),
                "supplier_engagement_pct": data.get("supplier_engagement_pct", 0),
                "data_quality_score": data.get("overall_dq_score", 0),
                "sbti_status": data.get("sbti_status", "Committed"),
                "cdp_score": data.get("cdp_score", "B"),
                "assurance_status": data.get("assurance_status", "Limited"),
                "carbon_price_applied": data.get("carbon_price", 0),
            },
            "rag_status": data.get("rag_indicators", {}),
            "initiatives": data.get("initiatives", []),
            "regulatory_compliance": data.get("regulatory_compliance", []),
            "peer_benchmarks": data.get("peer_benchmarks", {}),
            "next_quarter_priorities": data.get("next_quarter_priorities", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Executive Climate Dashboard\n\n"
            f"**{data.get('org_name', 'Enterprise')}** | "
            f"Q{data.get('quarter', '?')} {data.get('reporting_year', '')} | "
            f"Generated: {ts}\n\n---"
        )

    def _md_headline_kpis(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        employees = int(data.get("employees", 1))
        revenue = float(data.get("revenue", 1))
        currency = data.get("currency", "USD")

        return (
            f"## Headline KPIs\n\n"
            f"| KPI | Value | Unit | YoY |\n"
            f"|-----|------:|------|:---:|\n"
            f"| **Total Emissions** | **{_dec_comma(total)}** | **tCO2e** | {data.get('yoy_total', '-')} |\n"
            f"| Scope 1 | {_dec_comma(s1)} | tCO2e | {data.get('yoy_s1', '-')} |\n"
            f"| Scope 2 | {_dec_comma(s2)} | tCO2e | {data.get('yoy_s2', '-')} |\n"
            f"| Scope 3 | {_dec_comma(s3)} | tCO2e | {data.get('yoy_s3', '-')} |\n"
            f"| Per Employee | {_dec(_safe_div(total, employees))} | tCO2e/FTE | - |\n"
            f"| Per {currency}1M Revenue | {_dec(_safe_div(total, revenue) * 1_000_000)} | tCO2e/{currency}1M | - |\n"
            f"| Renewable Energy | {_pct(data.get('renewable_energy_pct', 0))} | of total electricity | - |\n"
            f"| Data Quality Score | {data.get('overall_dq_score', 0)} | /100 | - |\n"
            f"| Carbon Price | ${data.get('carbon_price', 0)}/tCO2e | Internal | - |"
        )

    def _md_target_progress(self, data: Dict[str, Any]) -> str:
        tp = data.get("target_progress_pct", 0)
        actual = float(data.get("actual_ytd_tco2e", 0))
        target = float(data.get("target_ytd_tco2e", 0))
        status = "ON TRACK" if actual <= target else "BEHIND"
        return (
            f"## Target Pathway Progress\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| SBTi Target Progress | {_pct(tp)} of milestone |\n"
            f"| YTD Actual | {_dec_comma(actual)} tCO2e |\n"
            f"| YTD Target | {_dec_comma(target)} tCO2e |\n"
            f"| Status | **{status}** |\n"
            f"| SBTi Validation | {data.get('sbti_status', 'Committed')} |\n"
            f"| CDP Score | {data.get('cdp_score', 'B')} |"
        )

    def _md_rag_indicators(self, data: Dict[str, Any]) -> str:
        indicators = data.get("rag_indicators", {
            "Scope 1 Reduction": "green",
            "Scope 2 (RE Procurement)": "green",
            "Scope 3 Data Quality": "amber",
            "Supplier Engagement": "amber",
            "SBTi Target Adherence": "green",
            "Regulatory Compliance": "green",
            "Board Reporting": "green",
            "Assurance Readiness": "amber",
        })
        lines = [
            "## RAG Status Indicators\n",
            "| Area | Status | Commentary |",
            "|------|:------:|------------|",
        ]
        commentaries = data.get("rag_commentaries", {})
        for area, status in indicators.items():
            commentary = commentaries.get(area, "")
            lines.append(f"| {area} | {_rag_md(status)} | {commentary} |")
        return "\n".join(lines)

    def _md_initiative_status(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        if not initiatives:
            return "## Key Initiatives\n\nNo initiatives tracked this quarter."
        lines = [
            "## Key Initiatives\n",
            "| Initiative | Category | Status | tCO2e Saved | Investment | Completion |",
            "|------------|----------|:------:|------------:|-----------:|:----------:|",
        ]
        for init in initiatives:
            lines.append(
                f"| {init.get('name', '')} | {init.get('category', '')} "
                f"| {_rag_md(init.get('status', 'amber'))} "
                f"| {_dec_comma(init.get('tco2e_saved', 0))} "
                f"| {init.get('currency', '$')}{_dec_comma(init.get('investment', 0))} "
                f"| {_pct(init.get('completion_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_regulatory_compliance(self, data: Dict[str, Any]) -> str:
        regs = data.get("regulatory_compliance", [
            {"framework": "SEC Climate Rule", "status": "green", "deadline": "Annual filing", "notes": "Scope 1+2 disclosed"},
            {"framework": "CSRD / ESRS E1", "status": "green", "deadline": "FY2025", "notes": "E1-1 through E1-9 prepared"},
            {"framework": "CDP Questionnaire", "status": "amber", "deadline": "July 2026", "notes": "In progress"},
            {"framework": "SBTi Progress", "status": "green", "deadline": "Annual", "notes": "On track"},
            {"framework": "CA SB 253", "status": "green", "deadline": "2027 (Scope 3)", "notes": "Scope 1+2 submitted"},
            {"framework": "ISO 14064-1", "status": "green", "deadline": "Annual", "notes": "Verified"},
        ])
        lines = [
            "## Regulatory Compliance\n",
            "| Framework | Status | Deadline | Notes |",
            "|-----------|:------:|----------|-------|",
        ]
        for reg in regs:
            lines.append(
                f"| {reg.get('framework', '')} | {_rag_md(reg.get('status', 'amber'))} "
                f"| {reg.get('deadline', '')} | {reg.get('notes', '')} |"
            )
        return "\n".join(lines)

    def _md_peer_benchmark(self, data: Dict[str, Any]) -> str:
        benchmarks = data.get("peer_benchmarks", {})
        if not benchmarks:
            return "## Peer Benchmarking\n\nPeer data not available for this period."
        lines = [
            "## Peer Benchmarking\n",
            "| Metric | Your Value | Sector Average | Best in Sector | Quartile |",
            "|--------|:----------:|:--------------:|:--------------:|:--------:|",
        ]
        for metric, values in benchmarks.items():
            lines.append(
                f"| {metric} | {values.get('yours', '-')} "
                f"| {values.get('sector_avg', '-')} "
                f"| {values.get('best', '-')} "
                f"| {values.get('quartile', '-')} |"
            )
        return "\n".join(lines)

    def _md_risks_opportunities(self, data: Dict[str, Any]) -> str:
        risks = data.get("top_risks", [])
        opps = data.get("top_opportunities", [])
        lines = ["## Key Risks & Opportunities\n"]
        if risks:
            lines.append("### Top Risks\n")
            for i, r in enumerate(risks[:3], 1):
                lines.append(f"{i}. **{r.get('name', '')}** -- {r.get('description', '')} "
                             f"(Impact: {r.get('impact', 'Medium')})")
        if opps:
            lines.append("\n### Top Opportunities\n")
            for i, o in enumerate(opps[:3], 1):
                lines.append(f"{i}. **{o.get('name', '')}** -- {o.get('description', '')} "
                             f"(Value: {o.get('value', 'Medium')})")
        return "\n".join(lines)

    def _md_next_quarter(self, data: Dict[str, Any]) -> str:
        priorities = data.get("next_quarter_priorities", [
            "Complete annual GHG inventory data collection",
            "Submit CDP Climate Change questionnaire",
            "Launch supplier engagement wave 2",
            "Present scenario analysis to Board",
            "Prepare for external assurance engagement",
        ])
        lines = ["## Next Quarter Priorities\n"]
        for i, p in enumerate(priorities, 1):
            lines.append(f"{i}. {p}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*Board-level executive dashboard. SHA-256 provenance.*"
        )
