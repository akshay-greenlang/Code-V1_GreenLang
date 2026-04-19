# -*- coding: utf-8 -*-
"""
AnnualProgressReportTemplate - Annual progress report for PACK-029.

Renders an annual progress report covering SBTi annual disclosure fields,
actual vs target emissions comparison, variance analysis, RAG performance
scoring, initiative deployment status, forward-looking projections, and
assurance statement section. Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  SBTi Annual Disclosure Fields
    3.  Actual vs Target Emissions (All Scopes)
    4.  Variance Analysis Summary
    5.  RAG Performance Scoring
    6.  Initiative Deployment Status
    7.  Forward-Looking Projection (3 Years)
    8.  Year-over-Year Trend
    9.  Assurance Statement
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

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
_TEMPLATE_ID = "annual_progress_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

RAG_THRESHOLDS = {
    "green": {"label": "GREEN", "desc": "On track or ahead of target", "max_deviation": 5.0},
    "amber": {"label": "AMBER", "desc": "Slight deviation, corrective actions needed", "max_deviation": 15.0},
    "red": {"label": "RED", "desc": "Significant deviation, urgent action required", "max_deviation": 100.0},
}

SBTI_DISCLOSURE_FIELDS = [
    "Organization name", "SBTi target status (committed/validated)",
    "Base year and base year emissions", "Near-term target year and target",
    "Long-term target year and target", "Current year emissions (Scope 1, 2, 3)",
    "Progress against targets (% reduction achieved)",
    "Methodology used (ACA/SDA)", "Recalculation events",
    "Off-track disclosure and remediation plan",
]

XBRL_TAGS: Dict[str, str] = {
    "reporting_year": "gl:AnnualProgressReportingYear",
    "actual_emissions": "gl:AnnualActualEmissions",
    "target_emissions": "gl:AnnualTargetEmissions",
    "variance_pct": "gl:AnnualVariancePercentage",
    "rag_status": "gl:AnnualRAGStatus",
    "on_track": "gl:AnnualOnTrackStatus",
    "initiatives_deployed": "gl:InitiativesDeployed",
    "assurance_level": "gl:AssuranceLevel",
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

def _pct_change(current: Any, baseline: Any) -> Decimal:
    c = Decimal(str(current))
    b = Decimal(str(baseline))
    if b == 0:
        return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _variance(actual: float, target: float) -> Dict[str, Any]:
    diff = actual - target
    pct = (diff / target * 100) if target != 0 else 0
    if abs(pct) <= RAG_THRESHOLDS["green"]["max_deviation"]:
        rag = "GREEN"
    elif abs(pct) <= RAG_THRESHOLDS["amber"]["max_deviation"]:
        rag = "AMBER"
    else:
        rag = "RED"
    return {
        "actual": actual, "target": target, "diff": round(diff, 2),
        "diff_pct": round(pct, 2), "rag": rag,
        "on_track": actual <= target,
    }

def _rag_color(status: str) -> str:
    return {"GREEN": _SUCCESS, "AMBER": _WARN, "RED": _DANGER}.get(status, "#999")

class AnnualProgressReportTemplate:
    """
    Annual progress report template for PACK-029 Interim Targets Pack.

    Renders SBTi annual disclosure format with actual vs target comparison,
    variance analysis, RAG scoring, initiative tracking, 3-year projections,
    and assurance statement. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = AnnualProgressReportTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_year": 2025,
        ...     "baseline_year": 2022,
        ...     "baseline_emissions": 100000,
        ...     "actual_emissions": {"scope1": 25000, "scope2": 15000, "scope3": 40000},
        ...     "target_emissions": {"scope1": 24000, "scope2": 14000, "scope3": 38000},
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full annual progress report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_sbti_disclosure(data), self._md_actual_vs_target(data),
            self._md_variance_analysis(data), self._md_rag_scoring(data),
            self._md_initiatives(data), self._md_projections(data),
            self._md_yoy_trend(data), self._md_assurance(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render full annual progress report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_sbti_disclosure(data), self._html_actual_vs_target(data),
            self._html_variance_analysis(data), self._html_rag_scoring(data),
            self._html_initiatives(data), self._html_projections(data),
            self._html_yoy_trend(data), self._html_assurance(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Annual Progress Report - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured JSON."""
        self.generated_at = utcnow()
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        var = _variance(total_actual, total_target)
        scope_variances = {}
        for scope in set(list(actual.keys()) + list(target.keys())):
            a = float(actual.get(scope, 0))
            t = float(target.get(scope, 0))
            scope_variances[scope] = _variance(a, t)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "baseline": {
                "year": data.get("baseline_year", ""),
                "emissions": str(data.get("baseline_emissions", 0)),
            },
            "actual_emissions": actual,
            "target_emissions": target,
            "total_actual": str(total_actual),
            "total_target": str(total_target),
            "overall_variance": var,
            "scope_variances": scope_variances,
            "initiatives": data.get("initiatives", []),
            "projections": data.get("projections", []),
            "assurance": data.get("assurance", {}),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Annual Progress - {data.get('org_name', '')}",
                "author": "GreenLang PACK-029",
            },
        }

    # -- Helpers --

    def _get_variance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        return _variance(total_actual, total_target)

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Annual Progress Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 Interim Targets Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        var = self._get_variance(data)
        baseline_em = float(data.get("baseline_emissions", 0))
        reduction = float(_pct_change(var["actual"], baseline_em)) if baseline_em else 0
        lines = [
            "## 1. Executive Summary\n",
            "| KPI | Value |", "|-----|-------|",
            f"| Reporting Year | {data.get('reporting_year', '')} |",
            f"| Total Actual Emissions | {_dec_comma(var['actual'], 0)} tCO2e |",
            f"| Total Target Emissions | {_dec_comma(var['target'], 0)} tCO2e |",
            f"| Variance | {'+' if var['diff'] > 0 else ''}{_dec_comma(var['diff'], 0)} tCO2e ({'+' if var['diff_pct'] > 0 else ''}{_dec(var['diff_pct'])}%) |",
            f"| RAG Status | **{var['rag']}** |",
            f"| On Track | **{'Yes' if var['on_track'] else 'No'}** |",
            f"| Reduction from Baseline | {_dec(abs(reduction))}% |",
        ]
        return "\n".join(lines)

    def _md_sbti_disclosure(self, data: Dict[str, Any]) -> str:
        disclosure = data.get("sbti_disclosure", {})
        lines = [
            "## 2. SBTi Annual Disclosure Fields\n",
            "| # | Field | Value |",
            "|---|-------|-------|",
        ]
        for i, field in enumerate(SBTI_DISCLOSURE_FIELDS, 1):
            key = field.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            val = disclosure.get(key, data.get(key, "-"))
            lines.append(f"| {i} | {field} | {val} |")
        return "\n".join(lines)

    def _md_actual_vs_target(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys())))
        lines = [
            "## 3. Actual vs Target Emissions\n",
            "| Scope | Actual (tCO2e) | Target (tCO2e) | Variance (tCO2e) | Variance (%) | Status |",
            "|-------|---------------:|---------------:|-----------------:|-------------:|--------|",
        ]
        total_a, total_t = 0.0, 0.0
        for scope in scopes:
            a = float(actual.get(scope, 0))
            t = float(target.get(scope, 0))
            v = _variance(a, t)
            total_a += a
            total_t += t
            lines.append(
                f"| {scope.replace('_', ' ').title()} | {_dec_comma(a, 0)} | {_dec_comma(t, 0)} "
                f"| {'+' if v['diff'] > 0 else ''}{_dec_comma(v['diff'], 0)} "
                f"| {'+' if v['diff_pct'] > 0 else ''}{_dec(v['diff_pct'])}% | **{v['rag']}** |"
            )
        total_v = _variance(total_a, total_t)
        lines.append(
            f"| **Total** | **{_dec_comma(total_a, 0)}** | **{_dec_comma(total_t, 0)}** "
            f"| **{'+' if total_v['diff'] > 0 else ''}{_dec_comma(total_v['diff'], 0)}** "
            f"| **{'+' if total_v['diff_pct'] > 0 else ''}{_dec(total_v['diff_pct'])}%** | **{total_v['rag']}** |"
        )
        return "\n".join(lines)

    def _md_variance_analysis(self, data: Dict[str, Any]) -> str:
        variance_detail = data.get("variance_detail", {})
        lines = [
            "## 4. Variance Analysis Summary\n",
            "| Factor | Impact (tCO2e) | Impact (%) | Direction |",
            "|--------|---------------:|-----------:|-----------|",
        ]
        factors = variance_detail.get("factors", [])
        for f in factors:
            impact = float(f.get("impact_tco2e", 0))
            direction = "Increase" if impact > 0 else "Decrease"
            lines.append(
                f"| {f.get('name', '')} | {'+' if impact > 0 else ''}{_dec_comma(impact, 0)} "
                f"| {'+' if impact > 0 else ''}{_dec(f.get('impact_pct', 0))}% | {direction} |"
            )
        if not factors:
            lines.append("| _No variance factors provided_ | - | - | - |")
        return "\n".join(lines)

    def _md_rag_scoring(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys())))
        lines = [
            "## 5. RAG Performance Scoring\n",
            "| Scope | Score | Status | Action Required |",
            "|-------|-------|--------|----------------|",
        ]
        for scope in scopes:
            v = _variance(float(actual.get(scope, 0)), float(target.get(scope, 0)))
            action = {
                "GREEN": "Continue monitoring",
                "AMBER": "Review and adjust initiatives",
                "RED": "Urgent corrective action required",
            }.get(v["rag"], "Review")
            lines.append(
                f"| {scope.replace('_', ' ').title()} | {_dec(abs(v['diff_pct']))}% deviation | **{v['rag']}** | {action} |"
            )
        lines.extend([
            "\n### RAG Definitions\n",
            f"| Status | Threshold | Description |",
            f"|--------|-----------|-------------|",
            f"| GREEN | <= {RAG_THRESHOLDS['green']['max_deviation']}% deviation | {RAG_THRESHOLDS['green']['desc']} |",
            f"| AMBER | <= {RAG_THRESHOLDS['amber']['max_deviation']}% deviation | {RAG_THRESHOLDS['amber']['desc']} |",
            f"| RED | > {RAG_THRESHOLDS['amber']['max_deviation']}% deviation | {RAG_THRESHOLDS['red']['desc']} |",
        ])
        return "\n".join(lines)

    def _md_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        lines = [
            "## 6. Initiative Deployment Status\n",
            "| # | Initiative | Status | Reduction (tCO2e) | Investment | Start |",
            "|---|-----------|--------|------------------:|------------|-------|",
        ]
        for i, init in enumerate(initiatives, 1):
            lines.append(
                f"| {i} | {init.get('name', '')} | {init.get('status', 'Planned')} "
                f"| {_dec_comma(init.get('reduction_tco2e', 0), 0)} "
                f"| {init.get('investment', 'N/A')} | {init.get('start_date', 'TBD')} |"
            )
        if not initiatives:
            lines.append("| - | _No initiatives reported_ | - | - | - | - |")
        total_reduction = sum(float(i.get("reduction_tco2e", 0)) for i in initiatives)
        lines.append(f"\n**Total Initiative Reduction:** {_dec_comma(total_reduction, 0)} tCO2e")
        return "\n".join(lines)

    def _md_projections(self, data: Dict[str, Any]) -> str:
        projections = data.get("projections", [])
        lines = [
            "## 7. Forward-Looking Projection (Next 3 Years)\n",
            "| Year | Projected Emissions (tCO2e) | Target (tCO2e) | Gap (tCO2e) | Confidence |",
            "|------|----------------------------:|---------------:|------------:|------------|",
        ]
        for p in projections[:3]:
            gap = float(p.get("projected", 0)) - float(p.get("target", 0))
            lines.append(
                f"| {p.get('year', '')} | {_dec_comma(p.get('projected', 0), 0)} "
                f"| {_dec_comma(p.get('target', 0), 0)} "
                f"| {'+' if gap > 0 else ''}{_dec_comma(gap, 0)} "
                f"| {p.get('confidence', 'Medium')} |"
            )
        if not projections:
            lines.append("| - | _No projections provided_ | - | - | - |")
        return "\n".join(lines)

    def _md_yoy_trend(self, data: Dict[str, Any]) -> str:
        history = data.get("historical_emissions", [])
        lines = [
            "## 8. Year-over-Year Trend\n",
            "| Year | Emissions (tCO2e) | YoY Change | YoY Change (%) |",
            "|------|------------------:|-----------:|---------------:|",
        ]
        prev = None
        for h in history:
            em = float(h.get("emissions", 0))
            if prev is not None:
                diff = em - prev
                pct = float(_pct_change(em, prev))
                lines.append(
                    f"| {h.get('year', '')} | {_dec_comma(em, 0)} "
                    f"| {'+' if diff > 0 else ''}{_dec_comma(diff, 0)} "
                    f"| {'+' if pct > 0 else ''}{_dec(pct)}% |"
                )
            else:
                lines.append(f"| {h.get('year', '')} | {_dec_comma(em, 0)} | - | - |")
            prev = em
        if not history:
            lines.append("| - | _No historical data_ | - | - |")
        return "\n".join(lines)

    def _md_assurance(self, data: Dict[str, Any]) -> str:
        assurance = data.get("assurance", {})
        lines = [
            "## 9. Assurance Statement\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Assurance Obtained | {assurance.get('obtained', 'No')} |",
            f"| Assurance Level | {assurance.get('level', 'N/A')} |",
            f"| Assurance Provider | {assurance.get('provider', 'N/A')} |",
            f"| Standard | {assurance.get('standard', 'ISAE 3410 / ISO 14064-3')} |",
            f"| Scope of Assurance | {assurance.get('scope', 'Scope 1+2')} |",
            f"| Opinion Type | {assurance.get('opinion', 'N/A')} |",
            f"| Material Misstatement | {assurance.get('material_misstatement', 'None identified')} |",
        ]
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        var = self._get_variance(data)
        lines = [
            "## 10. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Reporting Year | {XBRL_TAGS['reporting_year']} | {data.get('reporting_year', '')} |",
            f"| Actual Emissions | {XBRL_TAGS['actual_emissions']} | {_dec_comma(var['actual'], 0)} tCO2e |",
            f"| Target Emissions | {XBRL_TAGS['target_emissions']} | {_dec_comma(var['target'], 0)} tCO2e |",
            f"| Variance | {XBRL_TAGS['variance_pct']} | {_dec(var['diff_pct'])}% |",
            f"| RAG Status | {XBRL_TAGS['rag_status']} | {var['rag']} |",
            f"| On Track | {XBRL_TAGS['on_track']} | {'Yes' if var['on_track'] else 'No'} |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 11. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*Annual progress report with SBTi disclosure format.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".rag-green{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".rag-amber{{background:#fff3e0;color:#e65100;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".rag-red{{background:#ffcdd2;color:#c62828;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Annual Progress Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name","")} | <strong>Year:</strong> {data.get("reporting_year","")} | <strong>Generated:</strong> {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        var = self._get_variance(data)
        rag_cls = f"rag-{var['rag'].lower()}"
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Actual</div><div class="card-value">{_dec_comma(var["actual"], 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Target</div><div class="card-value">{_dec_comma(var["target"], 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Variance</div><div class="card-value">{_dec(var["diff_pct"])}%</div><div class="card-unit"><span class="{rag_cls}">{var["rag"]}</span></div></div>\n'
            f'<div class="card"><div class="card-label">On Track</div><div class="card-value">{"Yes" if var["on_track"] else "No"}</div></div>\n'
            f'</div>'
        )

    def _html_sbti_disclosure(self, data: Dict[str, Any]) -> str:
        disclosure = data.get("sbti_disclosure", {})
        rows = ""
        for i, field in enumerate(SBTI_DISCLOSURE_FIELDS, 1):
            key = field.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            val = disclosure.get(key, data.get(key, "-"))
            rows += f'<tr><td>{i}</td><td>{field}</td><td>{val}</td></tr>\n'
        return f'<h2>2. SBTi Disclosure</h2>\n<table>\n<tr><th>#</th><th>Field</th><th>Value</th></tr>\n{rows}</table>'

    def _html_actual_vs_target(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys())))
        rows = ""
        for scope in scopes:
            v = _variance(float(actual.get(scope, 0)), float(target.get(scope, 0)))
            rag_cls = f"rag-{v['rag'].lower()}"
            rows += (
                f'<tr><td>{scope.replace("_", " ").title()}</td>'
                f'<td>{_dec_comma(v["actual"], 0)}</td><td>{_dec_comma(v["target"], 0)}</td>'
                f'<td>{_dec_comma(v["diff"], 0)}</td><td>{_dec(v["diff_pct"])}%</td>'
                f'<td><span class="{rag_cls}">{v["rag"]}</span></td></tr>\n'
            )
        return (
            f'<h2>3. Actual vs Target</h2>\n<table>\n'
            f'<tr><th>Scope</th><th>Actual</th><th>Target</th><th>Variance</th><th>%</th><th>RAG</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_variance_analysis(self, data: Dict[str, Any]) -> str:
        factors = data.get("variance_detail", {}).get("factors", [])
        rows = ""
        for f in factors:
            impact = float(f.get("impact_tco2e", 0))
            rows += f'<tr><td>{f.get("name","")}</td><td>{_dec_comma(impact, 0)}</td><td>{_dec(f.get("impact_pct",0))}%</td></tr>\n'
        return f'<h2>4. Variance Analysis</h2>\n<table>\n<tr><th>Factor</th><th>Impact (tCO2e)</th><th>Impact (%)</th></tr>\n{rows}</table>'

    def _html_rag_scoring(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys())))
        rows = ""
        for scope in scopes:
            v = _variance(float(actual.get(scope, 0)), float(target.get(scope, 0)))
            rag_cls = f"rag-{v['rag'].lower()}"
            rows += f'<tr><td>{scope.replace("_"," ").title()}</td><td>{_dec(abs(v["diff_pct"]))}%</td><td><span class="{rag_cls}">{v["rag"]}</span></td></tr>\n'
        return f'<h2>5. RAG Scoring</h2>\n<table>\n<tr><th>Scope</th><th>Deviation</th><th>Status</th></tr>\n{rows}</table>'

    def _html_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        rows = ""
        for i, init in enumerate(initiatives, 1):
            rows += f'<tr><td>{i}</td><td>{init.get("name","")}</td><td>{init.get("status","")}</td><td>{_dec_comma(init.get("reduction_tco2e",0), 0)}</td></tr>\n'
        return f'<h2>6. Initiatives</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>Status</th><th>Reduction (tCO2e)</th></tr>\n{rows}</table>'

    def _html_projections(self, data: Dict[str, Any]) -> str:
        projections = data.get("projections", [])
        rows = ""
        for p in projections[:3]:
            rows += f'<tr><td>{p.get("year","")}</td><td>{_dec_comma(p.get("projected",0), 0)}</td><td>{_dec_comma(p.get("target",0), 0)}</td></tr>\n'
        return f'<h2>7. Projections</h2>\n<table>\n<tr><th>Year</th><th>Projected</th><th>Target</th></tr>\n{rows}</table>'

    def _html_yoy_trend(self, data: Dict[str, Any]) -> str:
        history = data.get("historical_emissions", [])
        rows = ""
        prev = None
        for h in history:
            em = float(h.get("emissions", 0))
            if prev is not None:
                pct = float(_pct_change(em, prev))
                rows += f'<tr><td>{h.get("year","")}</td><td>{_dec_comma(em, 0)}</td><td>{_dec(pct)}%</td></tr>\n'
            else:
                rows += f'<tr><td>{h.get("year","")}</td><td>{_dec_comma(em, 0)}</td><td>-</td></tr>\n'
            prev = em
        return f'<h2>8. YoY Trend</h2>\n<table>\n<tr><th>Year</th><th>Emissions</th><th>YoY %</th></tr>\n{rows}</table>'

    def _html_assurance(self, data: Dict[str, Any]) -> str:
        a = data.get("assurance", {})
        return (
            f'<h2>9. Assurance</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Obtained</td><td>{a.get("obtained","No")}</td></tr>\n'
            f'<tr><td>Level</td><td>{a.get("level","N/A")}</td></tr>\n'
            f'<tr><td>Provider</td><td>{a.get("provider","N/A")}</td></tr>\n'
            f'<tr><td>Standard</td><td>{a.get("standard","ISAE 3410")}</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>10. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>11. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - Annual progress report</div>'
