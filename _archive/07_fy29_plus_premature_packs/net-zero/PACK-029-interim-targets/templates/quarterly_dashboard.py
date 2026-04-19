# -*- coding: utf-8 -*-
"""
QuarterlyDashboardTemplate - Board-level quarterly KPI dashboard for PACK-029.

Renders a 1-page executive quarterly dashboard with key metrics, RAG alerts,
top performing initiatives, top risks/issues, and next quarter milestones.
Designed for board and C-suite consumption with concise visual layout.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Dashboard Header (Quarter, Period)
    2.  Key Metrics (Actual, Target, Variance, % Complete)
    3.  RAG Milestone Achievement
    4.  Top 5 Performing Initiatives
    5.  Top 3 Risks / Issues
    6.  Quarterly Trend (4-Quarter Rolling)
    7.  Next Quarter Milestones
    8.  XBRL Tagging Summary
    9.  Audit Trail & Provenance

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
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

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "quarterly_dashboard"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

XBRL_TAGS: Dict[str, str] = {
    "quarter": "gl:QuarterlyDashboardPeriod",
    "actual_emissions_q": "gl:QuarterlyActualEmissions",
    "target_emissions_q": "gl:QuarterlyTargetEmissions",
    "variance_q": "gl:QuarterlyVariance",
    "pct_complete": "gl:AnnualTargetCompletionPct",
    "rag_overall": "gl:QuarterlyRAGStatus",
}

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)

def _rag_from_variance(variance_pct: float) -> str:
    if abs(variance_pct) <= 5.0:
        return "GREEN"
    elif abs(variance_pct) <= 15.0:
        return "AMBER"
    return "RED"

class QuarterlyDashboardTemplate:
    """
    Board-level quarterly KPI dashboard template for PACK-029.

    Renders a concise 1-page executive dashboard with key emissions metrics,
    RAG status, top initiatives, top risks, and next quarter milestones.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = QuarterlyDashboardTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "quarter": "Q2 2025",
        ...     "actual_emissions_q": 22000,
        ...     "target_emissions_q": 21000,
        ...     "actual_emissions_ytd": 44000,
        ...     "target_emissions_ytd": 42000,
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render quarterly dashboard as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_key_metrics(data),
            self._md_rag_milestones(data), self._md_top_initiatives(data),
            self._md_top_risks(data), self._md_quarterly_trend(data),
            self._md_next_quarter(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render quarterly dashboard as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_key_metrics(data),
            self._html_rag_milestones(data), self._html_top_initiatives(data),
            self._html_top_risks(data), self._html_quarterly_trend(data),
            self._html_next_quarter(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Quarterly Dashboard - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        actual_q = float(data.get("actual_emissions_q", 0))
        target_q = float(data.get("target_emissions_q", 0))
        variance_q = actual_q - target_q
        var_pct = (variance_q / target_q * 100) if target_q else 0
        actual_ytd = float(data.get("actual_emissions_ytd", 0))
        target_ytd = float(data.get("target_emissions_ytd", 0))
        annual_target = float(data.get("annual_target", target_ytd * 4 if target_ytd else 0))
        pct_complete = (actual_ytd / annual_target * 100) if annual_target else 0

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "quarter": data.get("quarter", ""),
            "metrics": {
                "actual_q": str(actual_q), "target_q": str(target_q),
                "variance_q": str(variance_q), "variance_pct": str(round(var_pct, 2)),
                "actual_ytd": str(actual_ytd), "target_ytd": str(target_ytd),
                "annual_target": str(annual_target),
                "pct_complete": str(round(pct_complete, 2)),
                "rag": _rag_from_variance(var_pct),
            },
            "milestones": data.get("milestones", []),
            "top_initiatives": data.get("top_initiatives", [])[:5],
            "top_risks": data.get("top_risks", [])[:3],
            "quarterly_trend": data.get("quarterly_trend", []),
            "next_quarter": data.get("next_quarter_milestones", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Quarterly Dashboard - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Quarterly Dashboard\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Quarter:** {data.get('quarter', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}\n\n---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        actual_q = float(data.get("actual_emissions_q", 0))
        target_q = float(data.get("target_emissions_q", 0))
        variance_q = actual_q - target_q
        var_pct = (variance_q / target_q * 100) if target_q else 0
        actual_ytd = float(data.get("actual_emissions_ytd", 0))
        target_ytd = float(data.get("target_emissions_ytd", 0))
        annual_target = float(data.get("annual_target", 0))
        pct_complete = (actual_ytd / annual_target * 100) if annual_target else 0
        rag = _rag_from_variance(var_pct)
        baseline_em = float(data.get("baseline_emissions", 0))
        reduction = ((baseline_em - actual_ytd) / baseline_em * 100) if baseline_em > 0 else 0
        lines = [
            "## 1. Key Metrics\n",
            "| KPI | This Quarter | YTD | Annual Target |",
            "|-----|:------------:|:---:|:-------------:|",
            f"| Actual Emissions | {_dec_comma(actual_q, 0)} tCO2e | {_dec_comma(actual_ytd, 0)} tCO2e | - |",
            f"| Target Emissions | {_dec_comma(target_q, 0)} tCO2e | {_dec_comma(target_ytd, 0)} tCO2e | {_dec_comma(annual_target, 0)} tCO2e |",
            f"| Variance | {'+' if variance_q > 0 else ''}{_dec_comma(variance_q, 0)} tCO2e | - | - |",
            f"| Variance (%) | {'+' if var_pct > 0 else ''}{_dec(var_pct)}% | - | - |",
            f"| % Complete (YTD vs Annual) | - | {_dec(pct_complete)}% | 100% |",
            f"| Reduction from Baseline | - | {_dec(reduction)}% | - |",
            f"| **Overall RAG** | **{rag}** | - | - |",
        ]
        return "\n".join(lines)

    def _md_rag_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 2. RAG Milestone Achievement\n",
            "| # | Milestone | Target Date | Status | RAG |",
            "|---|-----------|-------------|--------|-----|",
        ]
        for i, m in enumerate(milestones, 1):
            lines.append(
                f"| {i} | {m.get('milestone', '')} | {m.get('target_date', '')} "
                f"| {m.get('status', 'Pending')} | **{m.get('rag', 'AMBER')}** |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - |")
        return "\n".join(lines)

    def _md_top_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("top_initiatives", [])[:5]
        lines = [
            "## 3. Top 5 Performing Initiatives\n",
            "| Rank | Initiative | Reduction (tCO2e) | Status | Performance |",
            "|------|-----------|------------------:|--------|-------------|",
        ]
        for i, init in enumerate(initiatives, 1):
            lines.append(
                f"| {i} | {init.get('name', '')} | {_dec_comma(init.get('reduction_tco2e', 0), 0)} "
                f"| {init.get('status', '')} | {init.get('performance', '')} |"
            )
        if not initiatives:
            lines.append("| - | _No initiatives reported_ | - | - | - |")
        return "\n".join(lines)

    def _md_top_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("top_risks", [])[:3]
        lines = [
            "## 4. Top 3 Risks / Issues\n",
            "| # | Risk | Impact | Likelihood | Mitigation | Owner |",
            "|---|------|--------|:----------:|-----------|-------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '')} | {r.get('impact', 'High')} "
                f"| {r.get('likelihood', 'Medium')} | {r.get('mitigation', '')} "
                f"| {r.get('owner', 'TBD')} |"
            )
        if not risks:
            lines.append("| - | _No risks flagged_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_quarterly_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("quarterly_trend", [])
        lines = [
            "## 5. Quarterly Trend (Rolling 4 Quarters)\n",
            "| Quarter | Actual (tCO2e) | Target (tCO2e) | Variance (%) | RAG |",
            "|---------|---------------:|---------------:|-------------:|-----|",
        ]
        for t in trend[-4:]:
            actual = float(t.get("actual", 0))
            target = float(t.get("target", 0))
            var = ((actual - target) / target * 100) if target else 0
            rag = _rag_from_variance(var)
            lines.append(
                f"| {t.get('quarter', '')} | {_dec_comma(actual, 0)} | {_dec_comma(target, 0)} "
                f"| {'+' if var > 0 else ''}{_dec(var)}% | **{rag}** |"
            )
        if not trend:
            lines.append("| - | _No trend data_ | - | - | - |")
        return "\n".join(lines)

    def _md_next_quarter(self, data: Dict[str, Any]) -> str:
        nq = data.get("next_quarter_milestones", [])
        lines = [
            "## 6. Next Quarter Milestones\n",
            "| # | Milestone | Target Date | Owner | Priority |",
            "|---|-----------|-------------|-------|----------|",
        ]
        for i, m in enumerate(nq, 1):
            lines.append(
                f"| {i} | {m.get('milestone', '')} | {m.get('target_date', '')} "
                f"| {m.get('owner', 'TBD')} | {m.get('priority', 'Medium')} |"
            )
        if not nq:
            lines.append("| - | _No milestones planned_ | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 7. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | - |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 8. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*Quarterly executive dashboard.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:30px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:18px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.75em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.7em;color:{_ACCENT};}}"
            f".rag-green{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:3px 10px;border-radius:4px;display:inline-block;}}"
            f".rag-amber{{background:#fff3e0;color:#e65100;font-weight:700;padding:3px 10px;border-radius:4px;display:inline-block;}}"
            f".rag-red{{background:#ffcdd2;color:#c62828;font-weight:700;padding:3px 10px;border-radius:4px;display:inline-block;}}"
            f".footer{{margin-top:30px;padding-top:15px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Quarterly Dashboard - {data.get("quarter","")}</h1>\n<p><strong>{data.get("org_name","")}</strong> | {ts}</p>'

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        actual_q = float(data.get("actual_emissions_q", 0))
        target_q = float(data.get("target_emissions_q", 0))
        var = actual_q - target_q
        var_pct = (var / target_q * 100) if target_q else 0
        rag = _rag_from_variance(var_pct)
        rag_cls = f"rag-{rag.lower()}"
        actual_ytd = float(data.get("actual_emissions_ytd", 0))
        return (
            f'<h2>1. Key Metrics</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Actual (Q)</div><div class="card-value">{_dec_comma(actual_q, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Target (Q)</div><div class="card-value">{_dec_comma(target_q, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Variance</div><div class="card-value">{_dec(var_pct)}%</div><div class="card-unit"><span class="{rag_cls}">{rag}</span></div></div>\n'
            f'<div class="card"><div class="card-label">YTD Actual</div><div class="card-value">{_dec_comma(actual_ytd, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_rag_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for i, m in enumerate(milestones, 1):
            rag = m.get("rag", "AMBER")
            rag_cls = f"rag-{rag.lower()}"
            rows += f'<tr><td>{i}</td><td>{m.get("milestone","")}</td><td>{m.get("target_date","")}</td><td>{m.get("status","")}</td><td><span class="{rag_cls}">{rag}</span></td></tr>\n'
        return f'<h2>2. Milestones</h2>\n<table>\n<tr><th>#</th><th>Milestone</th><th>Date</th><th>Status</th><th>RAG</th></tr>\n{rows}</table>'

    def _html_top_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("top_initiatives", [])[:5]
        rows = ""
        for i, init in enumerate(initiatives, 1):
            rows += f'<tr><td>{i}</td><td>{init.get("name","")}</td><td>{_dec_comma(init.get("reduction_tco2e",0), 0)}</td><td>{init.get("status","")}</td></tr>\n'
        return f'<h2>3. Top Initiatives</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>Reduction</th><th>Status</th></tr>\n{rows}</table>'

    def _html_top_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("top_risks", [])[:3]
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("risk","")}</td><td>{r.get("impact","")}</td><td>{r.get("mitigation","")}</td></tr>\n'
        return f'<h2>4. Top Risks</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'

    def _html_quarterly_trend(self, data: Dict[str, Any]) -> str:
        trend = data.get("quarterly_trend", [])
        rows = ""
        for t in trend[-4:]:
            actual = float(t.get("actual", 0))
            target = float(t.get("target", 0))
            var = ((actual - target) / target * 100) if target else 0
            rag = _rag_from_variance(var)
            rag_cls = f"rag-{rag.lower()}"
            rows += f'<tr><td>{t.get("quarter","")}</td><td>{_dec_comma(actual, 0)}</td><td>{_dec_comma(target, 0)}</td><td>{_dec(var)}%</td><td><span class="{rag_cls}">{rag}</span></td></tr>\n'
        return f'<h2>5. Quarterly Trend</h2>\n<table>\n<tr><th>Quarter</th><th>Actual</th><th>Target</th><th>Var %</th><th>RAG</th></tr>\n{rows}</table>'

    def _html_next_quarter(self, data: Dict[str, Any]) -> str:
        nq = data.get("next_quarter_milestones", [])
        rows = ""
        for i, m in enumerate(nq, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("milestone","")}</td><td>{m.get("target_date","")}</td><td>{m.get("owner","TBD")}</td></tr>\n'
        return f'<h2>6. Next Quarter</h2>\n<table>\n<tr><th>#</th><th>Milestone</th><th>Date</th><th>Owner</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>7. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>8. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - Quarterly dashboard</div>'
