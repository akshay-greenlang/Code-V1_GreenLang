# -*- coding: utf-8 -*-
"""
CorrectiveActionPlanTemplate - Gap-to-target corrective action plan for PACK-029.

Renders a corrective action plan with gap-to-target quantification, candidate
initiative portfolio (MACC curve data), initiative scheduling, investment
requirements, risk assessment, and expected reduction impact per initiative.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Gap-to-Target Quantification
    3.  Candidate Initiative Portfolio (MACC)
    4.  Initiative Scheduling (Phased Timeline)
    5.  Investment Requirements (CapEx / OpEx)
    6.  Risk Assessment
    7.  Expected Reduction Impact
    8.  Implementation Roadmap
    9.  XBRL Tagging Summary
    10. Audit Trail & Provenance

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "corrective_action_plan"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

RISK_CATEGORIES = [
    {"id": "technology", "name": "Technology Risk", "desc": "Technology readiness and scalability"},
    {"id": "regulatory", "name": "Regulatory Risk", "desc": "Policy changes and compliance requirements"},
    {"id": "market", "name": "Market Risk", "desc": "Cost fluctuations and supply chain dependencies"},
    {"id": "execution", "name": "Execution Risk", "desc": "Implementation timeline and resource availability"},
    {"id": "financial", "name": "Financial Risk", "desc": "Funding availability and ROI uncertainty"},
]

INITIATIVE_PHASES = [
    {"id": "quick_wins", "name": "Phase 1: Quick Wins", "timeline": "0-12 months"},
    {"id": "medium_term", "name": "Phase 2: Medium-Term", "timeline": "1-3 years"},
    {"id": "strategic", "name": "Phase 3: Strategic", "timeline": "3-5+ years"},
]

XBRL_TAGS: Dict[str, str] = {
    "gap_to_target": "gl:GapToTargetEmissions",
    "total_abatement_potential": "gl:TotalAbatementPotential",
    "total_capex": "gl:CorrectiveActionTotalCapEx",
    "total_opex": "gl:CorrectiveActionTotalOpEx",
    "initiatives_count": "gl:CorrectiveInitiativesCount",
    "gap_closure_pct": "gl:GapClosurePercentage",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

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


class CorrectiveActionPlanTemplate:
    """
    Corrective action plan template for PACK-029 Interim Targets Pack.

    Renders gap-to-target analysis, MACC initiative portfolio, phased
    scheduling, investment requirements, risk assessment, and reduction
    impact per initiative. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = CorrectiveActionPlanTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "current_emissions": 92000,
        ...     "target_emissions": 80000,
        ...     "target_year": 2030,
        ...     "initiatives": [
        ...         {"name": "LED Retrofit", "reduction_tco2e": 3000, "cost_per_tco2e": -50},
        ...     ],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full corrective action plan as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_gap_quantification(data), self._md_macc(data),
            self._md_scheduling(data), self._md_investment(data),
            self._md_risk_assessment(data), self._md_reduction_impact(data),
            self._md_roadmap(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render full corrective action plan as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_gap_quantification(data), self._html_macc(data),
            self._html_scheduling(data), self._html_investment(data),
            self._html_risk_assessment(data), self._html_reduction_impact(data),
            self._html_roadmap(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Corrective Action Plan - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured JSON."""
        self.generated_at = _utcnow()
        current = float(data.get("current_emissions", 0))
        target = float(data.get("target_emissions", 0))
        gap = current - target
        initiatives = data.get("initiatives", [])
        total_abatement = sum(float(i.get("reduction_tco2e", 0)) for i in initiatives)
        gap_closure = (total_abatement / gap * 100) if gap > 0 else 0
        total_capex = sum(float(i.get("capex", 0)) for i in initiatives)
        total_opex = sum(float(i.get("annual_opex", 0)) for i in initiatives)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "gap_analysis": {
                "current_emissions": str(current), "target_emissions": str(target),
                "gap_tco2e": str(gap), "target_year": data.get("target_year", ""),
                "total_abatement_potential": str(total_abatement),
                "gap_closure_pct": str(round(gap_closure, 2)),
                "residual_gap": str(round(max(0, gap - total_abatement), 2)),
            },
            "initiatives": initiatives,
            "investment": {"total_capex": str(total_capex), "total_opex": str(total_opex)},
            "risks": data.get("risks", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Corrective Action Plan - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Helpers --

    def _get_gap(self, data: Dict[str, Any]) -> float:
        return float(data.get("current_emissions", 0)) - float(data.get("target_emissions", 0))

    def _get_total_abatement(self, data: Dict[str, Any]) -> float:
        return sum(float(i.get("reduction_tco2e", 0)) for i in data.get("initiatives", []))

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Corrective Action Plan\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Target Year:** {data.get('target_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 Interim Targets Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        gap = self._get_gap(data)
        total_abatement = self._get_total_abatement(data)
        gap_closure = (total_abatement / gap * 100) if gap > 0 else 0
        initiatives = data.get("initiatives", [])
        total_capex = sum(float(i.get("capex", 0)) for i in initiatives)
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Current Emissions | {_dec_comma(data.get('current_emissions', 0), 0)} tCO2e |",
            f"| Target Emissions | {_dec_comma(data.get('target_emissions', 0), 0)} tCO2e |",
            f"| Gap to Target | {_dec_comma(gap, 0)} tCO2e |",
            f"| Total Abatement Potential | {_dec_comma(total_abatement, 0)} tCO2e |",
            f"| Gap Closure | {_dec(gap_closure)}% |",
            f"| Residual Gap | {_dec_comma(max(0, gap - total_abatement), 0)} tCO2e |",
            f"| Initiatives Identified | {len(initiatives)} |",
            f"| Total CapEx Required | {_dec_comma(total_capex, 0)} |",
        ]
        return "\n".join(lines)

    def _md_gap_quantification(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_emissions", 0))
        target = float(data.get("target_emissions", 0))
        baseline = float(data.get("baseline_emissions", current))
        target_year = data.get("target_year", "")
        current_year = data.get("current_year", "")
        gap = current - target
        trajectory = data.get("current_trajectory", {})
        lines = [
            "## 2. Gap-to-Target Quantification\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Current Emissions ({current_year}) | {_dec_comma(current, 0)} tCO2e |",
            f"| Target Emissions ({target_year}) | {_dec_comma(target, 0)} tCO2e |",
            f"| Absolute Gap | {_dec_comma(gap, 0)} tCO2e |",
            f"| Gap as % of Baseline | {_dec((gap / baseline * 100) if baseline > 0 else 0)}% |",
            f"| Projected Trajectory | {trajectory.get('projected_emissions', 'N/A')} tCO2e by {target_year} |",
            f"| Trajectory vs Target Gap | {_dec_comma(trajectory.get('trajectory_gap', 0), 0)} tCO2e |",
        ]
        return "\n".join(lines)

    def _md_macc(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        # Sort by cost per tCO2e (MACC order: cheapest first)
        sorted_init = sorted(initiatives, key=lambda x: float(x.get("cost_per_tco2e", 0)))
        lines = [
            "## 3. Candidate Initiative Portfolio (MACC Curve)\n",
            "Initiatives sorted by Marginal Abatement Cost (cheapest first).\n",
            "| # | Initiative | Reduction (tCO2e) | Cost/tCO2e | Cumulative (tCO2e) | Phase |",
            "|---|-----------|------------------:|-----------:|-------------------:|-------|",
        ]
        cumulative = 0.0
        for i, init in enumerate(sorted_init, 1):
            reduction = float(init.get("reduction_tco2e", 0))
            cumulative += reduction
            cost = float(init.get("cost_per_tco2e", 0))
            cost_str = f"${_dec(cost)}" if cost >= 0 else f"-${_dec(abs(cost))} (saving)"
            lines.append(
                f"| {i} | {init.get('name', '')} | {_dec_comma(reduction, 0)} "
                f"| {cost_str} | {_dec_comma(cumulative, 0)} | {init.get('phase', '')} |"
            )
        gap = self._get_gap(data)
        lines.append(f"\n**Total Abatement:** {_dec_comma(cumulative, 0)} tCO2e "
                      f"({_dec((cumulative / gap * 100) if gap > 0 else 0)}% of gap)")
        return "\n".join(lines)

    def _md_scheduling(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        lines = [
            "## 4. Initiative Scheduling (Phased Deployment)\n",
        ]
        for phase_def in INITIATIVE_PHASES:
            phase_inits = [i for i in initiatives if i.get("phase") == phase_def["id"]]
            lines.append(f"\n### {phase_def['name']} ({phase_def['timeline']})\n")
            if phase_inits:
                lines.append("| Initiative | Start | End | Reduction (tCO2e) | Dependencies |")
                lines.append("|-----------|-------|-----|------------------:|-------------|")
                for init in phase_inits:
                    lines.append(
                        f"| {init.get('name', '')} | {init.get('start', 'TBD')} | {init.get('end', 'TBD')} "
                        f"| {_dec_comma(init.get('reduction_tco2e', 0), 0)} "
                        f"| {init.get('dependencies', 'None')} |"
                    )
            else:
                lines.append("_No initiatives in this phase._")
        return "\n".join(lines)

    def _md_investment(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        total_capex = sum(float(i.get("capex", 0)) for i in initiatives)
        total_opex = sum(float(i.get("annual_opex", 0)) for i in initiatives)
        lines = [
            "## 5. Investment Requirements\n",
            "| # | Initiative | CapEx | Annual OpEx | Payback (yrs) | NPV | ROI |",
            "|---|-----------|------:|----------:|:-------------:|----:|----:|",
        ]
        for i, init in enumerate(initiatives, 1):
            lines.append(
                f"| {i} | {init.get('name', '')} | {_dec_comma(init.get('capex', 0), 0)} "
                f"| {_dec_comma(init.get('annual_opex', 0), 0)} "
                f"| {init.get('payback_years', 'N/A')} "
                f"| {_dec_comma(init.get('npv', 0), 0)} "
                f"| {_dec(init.get('roi_pct', 0))}% |"
            )
        lines.extend([
            f"\n**Total CapEx:** {_dec_comma(total_capex, 0)}  \n"
            f"**Total Annual OpEx:** {_dec_comma(total_opex, 0)}",
        ])
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 6. Risk Assessment\n",
            "| # | Risk | Category | Likelihood | Impact | Mitigation |",
            "|---|------|----------|:----------:|:------:|-----------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '')} | {r.get('category', '')} "
                f"| {r.get('likelihood', 'Medium')} | {r.get('impact', 'Medium')} "
                f"| {r.get('mitigation', '')} |"
            )
        if not risks:
            lines.append("| - | _No risks assessed_ | - | - | - | - |")
        lines.extend(["\n### Risk Category Definitions\n"])
        for cat in RISK_CATEGORIES:
            lines.append(f"- **{cat['name']}:** {cat['desc']}")
        return "\n".join(lines)

    def _md_reduction_impact(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        gap = self._get_gap(data)
        lines = [
            "## 7. Expected Reduction Impact\n",
            "| # | Initiative | Reduction (tCO2e) | % of Gap | Confidence | Scope |",
            "|---|-----------|------------------:|---------:|:----------:|-------|",
        ]
        for i, init in enumerate(initiatives, 1):
            reduction = float(init.get("reduction_tco2e", 0))
            pct_gap = (reduction / gap * 100) if gap > 0 else 0
            lines.append(
                f"| {i} | {init.get('name', '')} | {_dec_comma(reduction, 0)} "
                f"| {_dec(pct_gap)}% | {init.get('confidence', 'Medium')} "
                f"| {init.get('scope', '')} |"
            )
        total = self._get_total_abatement(data)
        lines.append(
            f"| - | **Total** | **{_dec_comma(total, 0)}** "
            f"| **{_dec((total / gap * 100) if gap > 0 else 0)}%** | - | - |"
        )
        return "\n".join(lines)

    def _md_roadmap(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 8. Implementation Roadmap\n",
            "| # | Milestone | Date | Owner | Status | Linked Initiatives |",
            "|---|-----------|------|-------|--------|--------------------|",
        ]
        for i, m in enumerate(milestones, 1):
            lines.append(
                f"| {i} | {m.get('milestone', '')} | {m.get('date', 'TBD')} "
                f"| {m.get('owner', 'TBD')} | {m.get('status', 'Planned')} "
                f"| {m.get('linked_initiatives', '-')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        gap = self._get_gap(data)
        total_abatement = self._get_total_abatement(data)
        lines = [
            "## 9. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Gap to Target | {XBRL_TAGS['gap_to_target']} | {_dec_comma(gap, 0)} tCO2e |",
            f"| Total Abatement | {XBRL_TAGS['total_abatement_potential']} | {_dec_comma(total_abatement, 0)} tCO2e |",
            f"| Initiatives | {XBRL_TAGS['initiatives_count']} | {len(data.get('initiatives', []))} |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 10. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*Corrective action plan with MACC analysis.*"

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
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Corrective Action Plan</h1>\n<p><strong>Organization:</strong> {data.get("org_name","")} | <strong>Target Year:</strong> {data.get("target_year","")} | <strong>Generated:</strong> {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        gap = self._get_gap(data)
        total = self._get_total_abatement(data)
        closure = (total / gap * 100) if gap > 0 else 0
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Gap to Target</div><div class="card-value">{_dec_comma(gap, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Abatement</div><div class="card-value">{_dec_comma(total, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Gap Closure</div><div class="card-value">{_dec(closure)}%</div></div>\n'
            f'<div class="card"><div class="card-label">Initiatives</div><div class="card-value">{len(data.get("initiatives",[]))}</div></div>\n'
            f'</div>'
        )

    def _html_gap_quantification(self, data: Dict[str, Any]) -> str:
        current = float(data.get("current_emissions", 0))
        target = float(data.get("target_emissions", 0))
        gap = current - target
        return (
            f'<h2>2. Gap Quantification</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Current</td><td>{_dec_comma(current, 0)} tCO2e</td></tr>\n'
            f'<tr><td>Target</td><td>{_dec_comma(target, 0)} tCO2e</td></tr>\n'
            f'<tr><td>Gap</td><td>{_dec_comma(gap, 0)} tCO2e</td></tr>\n</table>'
        )

    def _html_macc(self, data: Dict[str, Any]) -> str:
        initiatives = sorted(data.get("initiatives", []), key=lambda x: float(x.get("cost_per_tco2e", 0)))
        rows = ""
        cumulative = 0.0
        for i, init in enumerate(initiatives, 1):
            red = float(init.get("reduction_tco2e", 0))
            cumulative += red
            rows += f'<tr><td>{i}</td><td>{init.get("name","")}</td><td>{_dec_comma(red, 0)}</td><td>${_dec(init.get("cost_per_tco2e",0))}</td><td>{_dec_comma(cumulative, 0)}</td></tr>\n'
        return f'<h2>3. MACC Curve</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>Reduction</th><th>Cost/tCO2e</th><th>Cumulative</th></tr>\n{rows}</table>'

    def _html_scheduling(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        rows = ""
        for ph in INITIATIVE_PHASES:
            phase_inits = [i for i in initiatives if i.get("phase") == ph["id"]]
            for init in phase_inits:
                rows += f'<tr><td>{ph["name"]}</td><td>{init.get("name","")}</td><td>{init.get("start","TBD")}</td><td>{init.get("end","TBD")}</td><td>{_dec_comma(init.get("reduction_tco2e",0), 0)}</td></tr>\n'
        return f'<h2>4. Scheduling</h2>\n<table>\n<tr><th>Phase</th><th>Initiative</th><th>Start</th><th>End</th><th>Reduction</th></tr>\n{rows}</table>'

    def _html_investment(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        rows = ""
        for i, init in enumerate(initiatives, 1):
            rows += f'<tr><td>{i}</td><td>{init.get("name","")}</td><td>{_dec_comma(init.get("capex",0), 0)}</td><td>{_dec_comma(init.get("annual_opex",0), 0)}</td><td>{init.get("payback_years","N/A")}</td></tr>\n'
        return f'<h2>5. Investment</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>CapEx</th><th>OpEx/yr</th><th>Payback</th></tr>\n{rows}</table>'

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("risk","")}</td><td>{r.get("category","")}</td><td>{r.get("likelihood","Medium")}</td><td>{r.get("impact","Medium")}</td></tr>\n'
        return f'<h2>6. Risks</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Category</th><th>Likelihood</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_reduction_impact(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        gap = self._get_gap(data)
        rows = ""
        for i, init in enumerate(initiatives, 1):
            red = float(init.get("reduction_tco2e", 0))
            pct = (red / gap * 100) if gap > 0 else 0
            rows += f'<tr><td>{i}</td><td>{init.get("name","")}</td><td>{_dec_comma(red, 0)}</td><td>{_dec(pct)}%</td></tr>\n'
        return f'<h2>7. Reduction Impact</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>Reduction</th><th>% of Gap</th></tr>\n{rows}</table>'

    def _html_roadmap(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for i, m in enumerate(milestones, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("milestone","")}</td><td>{m.get("date","TBD")}</td><td>{m.get("status","Planned")}</td></tr>\n'
        return f'<h2>8. Roadmap</h2>\n<table>\n<tr><th>#</th><th>Milestone</th><th>Date</th><th>Status</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>9. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>10. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - Corrective action plan</div>'
