# -*- coding: utf-8 -*-
"""
TCFDMetricsReportTemplate - TCFD Metrics & Targets pillar disclosure for PACK-029.

Renders a TCFD-aligned metrics and targets report covering GHG emissions
by scope with interim targets, transition risks and opportunities linked
to targets, forward-looking metrics, scenario analysis integration, and
TCFD recommendation alignment. Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  GHG Emissions by Scope with Interim Targets
    3.  Transition Risks Linked to Targets
    4.  Transition Opportunities Linked to Targets
    5.  Forward-Looking Metrics
    6.  Scenario Analysis Integration
    7.  Carbon Pricing Impact
    8.  Internal Carbon Price Application
    9.  TCFD Recommendation Alignment
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
_TEMPLATE_ID = "tcfd_metrics_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

TCFD_METRIC_RECS = [
    {"id": "ghg_scope1", "rec": "Scope 1 GHG emissions disclosed", "pillar": "Metrics"},
    {"id": "ghg_scope2", "rec": "Scope 2 GHG emissions disclosed", "pillar": "Metrics"},
    {"id": "ghg_scope3", "rec": "Scope 3 GHG emissions disclosed (if material)", "pillar": "Metrics"},
    {"id": "targets_interim", "rec": "Interim GHG reduction targets described", "pillar": "Targets"},
    {"id": "targets_netzero", "rec": "Net-zero target disclosed with timeline", "pillar": "Targets"},
    {"id": "targets_progress", "rec": "Progress against targets reported", "pillar": "Targets"},
    {"id": "scenario_analysis", "rec": "Scenario analysis results integrated", "pillar": "Strategy"},
    {"id": "carbon_price", "rec": "Internal carbon price disclosed", "pillar": "Metrics"},
    {"id": "transition_risks", "rec": "Transition risks linked to targets", "pillar": "Risk Mgmt"},
    {"id": "opportunities", "rec": "Climate opportunities quantified", "pillar": "Strategy"},
]

XBRL_TAGS: Dict[str, str] = {
    "scope1_emissions": "gl:TCFDScope1Emissions",
    "scope2_emissions": "gl:TCFDScope2Emissions",
    "scope3_emissions": "gl:TCFDScope3Emissions",
    "internal_carbon_price": "gl:TCFDInternalCarbonPrice",
    "scenario_used": "gl:TCFDScenarioUsed",
    "tcfd_alignment_score": "gl:TCFDAlignmentScore",
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

class TCFDMetricsReportTemplate:
    """
    TCFD Metrics and Targets pillar report template for PACK-029.

    Renders GHG emissions by scope with interim targets, transition risks
    and opportunities, forward-looking metrics, scenario analysis, carbon
    pricing impact, and TCFD alignment assessment. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = TCFDMetricsReportTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "emissions": {"scope1": 30000, "scope2": 20000, "scope3": 50000},
        ...     "targets": [{"scope": "1+2", "year": 2030, "reduction_pct": 46.2}],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render TCFD metrics report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_ghg_emissions(data), self._md_transition_risks(data),
            self._md_opportunities(data), self._md_forward_metrics(data),
            self._md_scenario_analysis(data), self._md_carbon_pricing(data),
            self._md_icp_application(data), self._md_tcfd_alignment(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render TCFD metrics report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_ghg_emissions(data), self._html_transition_risks(data),
            self._html_opportunities(data), self._html_forward_metrics(data),
            self._html_scenario_analysis(data), self._html_carbon_pricing(data),
            self._html_icp_application(data), self._html_tcfd_alignment(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>TCFD Metrics - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        emissions = data.get("emissions", {})
        alignment_results = data.get("tcfd_alignment", {})
        passed = sum(1 for v in alignment_results.values() if v.get("status") == "pass")
        total = len(TCFD_METRIC_RECS)
        score = (passed / max(1, total)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "emissions": emissions,
            "targets": data.get("targets", []),
            "transition_risks": data.get("transition_risks", []),
            "opportunities": data.get("opportunities", []),
            "forward_metrics": data.get("forward_metrics", {}),
            "scenario_analysis": data.get("scenario_analysis", {}),
            "carbon_pricing": data.get("carbon_pricing", {}),
            "tcfd_alignment": {"score": str(round(score, 1)), "passed": passed, "total": total},
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"TCFD Metrics - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# TCFD Metrics & Targets Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        em = data.get("emissions", {})
        total = sum(float(v) for v in em.values())
        targets = data.get("targets", [])
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Total GHG Emissions | {_dec_comma(total, 0)} tCO2e |",
            f"| Scope 1 | {_dec_comma(em.get('scope1', 0), 0)} tCO2e |",
            f"| Scope 2 | {_dec_comma(em.get('scope2', 0), 0)} tCO2e |",
            f"| Scope 3 | {_dec_comma(em.get('scope3', 0), 0)} tCO2e |",
            f"| Interim Targets | {len(targets)} |",
            f"| Internal Carbon Price | ${_dec(data.get('carbon_pricing', {}).get('internal_price', 0))}/tCO2e |",
        ]
        return "\n".join(lines)

    def _md_ghg_emissions(self, data: Dict[str, Any]) -> str:
        em = data.get("emissions", {})
        targets = data.get("targets", [])
        lines = [
            "## 2. GHG Emissions by Scope with Interim Targets\n",
            "| Scope | Current (tCO2e) | Target Year | Target Reduction | Target (tCO2e) |",
            "|-------|----------------:|:-----------:|-----------------:|---------------:|",
        ]
        for scope_key in ["scope1", "scope2", "scope3"]:
            current = float(em.get(scope_key, 0))
            matching = [t for t in targets if scope_key in t.get("scope", "").lower().replace(" ", "").replace("+", "")]
            if matching:
                t = matching[0]
                lines.append(
                    f"| {scope_key.replace('scope', 'Scope ').title()} | {_dec_comma(current, 0)} "
                    f"| {t.get('year', '')} | -{_dec(t.get('reduction_pct', 0))}% "
                    f"| {_dec_comma(current * (1 - float(t.get('reduction_pct', 0)) / 100), 0)} |"
                )
            else:
                lines.append(f"| {scope_key.replace('scope', 'Scope ').title()} | {_dec_comma(current, 0)} | - | - | - |")
        total = sum(float(v) for v in em.values())
        lines.append(f"| **Total** | **{_dec_comma(total, 0)}** | - | - | - |")
        return "\n".join(lines)

    def _md_transition_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("transition_risks", [])
        lines = [
            "## 3. Transition Risks Linked to Targets\n",
            "| # | Risk | Type | Financial Impact | Time Horizon | Linked Target |",
            "|---|------|------|:----------------:|:------------:|-------------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '')} | {r.get('type', 'Policy')} "
                f"| {r.get('financial_impact', 'Medium')} | {r.get('time_horizon', 'Medium-term')} "
                f"| {r.get('linked_target', '-')} |"
            )
        if not risks:
            lines.append("| - | _No transition risks identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        opportunities = data.get("opportunities", [])
        lines = [
            "## 4. Transition Opportunities Linked to Targets\n",
            "| # | Opportunity | Type | Financial Impact | Time Horizon | Linked Target |",
            "|---|-----------|------|:----------------:|:------------:|-------------|",
        ]
        for i, o in enumerate(opportunities, 1):
            lines.append(
                f"| {i} | {o.get('opportunity', '')} | {o.get('type', 'Resource Efficiency')} "
                f"| {o.get('financial_impact', 'Medium')} | {o.get('time_horizon', 'Medium-term')} "
                f"| {o.get('linked_target', '-')} |"
            )
        if not opportunities:
            lines.append("| - | _No opportunities identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_forward_metrics(self, data: Dict[str, Any]) -> str:
        fm = data.get("forward_metrics", {})
        projections = fm.get("projections", [])
        lines = [
            "## 5. Forward-Looking Metrics\n",
            "| Year | Projected Emissions (tCO2e) | Target (tCO2e) | Scenario | Confidence |",
            "|------|----------------------------:|---------------:|----------|:----------:|",
        ]
        for p in projections:
            lines.append(
                f"| {p.get('year', '')} | {_dec_comma(p.get('projected', 0), 0)} "
                f"| {_dec_comma(p.get('target', 0), 0)} "
                f"| {p.get('scenario', 'BAU')} | {p.get('confidence', 'Medium')} |"
            )
        if not projections:
            lines.append("| - | _No projections provided_ | - | - | - |")
        return "\n".join(lines)

    def _md_scenario_analysis(self, data: Dict[str, Any]) -> str:
        sa = data.get("scenario_analysis", {})
        scenarios = sa.get("scenarios", [])
        lines = [
            "## 6. Scenario Analysis Integration\n",
            "| Scenario | Temperature | Emissions Impact | Financial Impact | Probability |",
            "|----------|:----------:|:----------------:|:----------------:|:-----------:|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('name', '')} | {s.get('temperature', '')} "
                f"| {s.get('emissions_impact', '')} "
                f"| {s.get('financial_impact', '')} | {s.get('probability', '')} |"
            )
        if not scenarios:
            lines.append("| - | _No scenario analysis_ | - | - | - |")
        return "\n".join(lines)

    def _md_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("carbon_pricing", {})
        lines = [
            "## 7. Carbon Pricing Impact\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Current Carbon Price Exposure | ${_dec(cp.get('current_exposure', 0))}/tCO2e |",
            f"| Projected 2030 Carbon Price | ${_dec(cp.get('projected_2030', 0))}/tCO2e |",
            f"| Financial Impact (Current) | {_dec_comma(cp.get('financial_impact_current', 0), 0)} |",
            f"| Financial Impact (2030) | {_dec_comma(cp.get('financial_impact_2030', 0), 0)} |",
            f"| Jurisdictions Covered | {cp.get('jurisdictions', 'N/A')} |",
        ]
        return "\n".join(lines)

    def _md_icp_application(self, data: Dict[str, Any]) -> str:
        icp = data.get("carbon_pricing", {})
        lines = [
            "## 8. Internal Carbon Price Application\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Internal Carbon Price | ${_dec(icp.get('internal_price', 0))}/tCO2e |",
            f"| Price Type | {icp.get('price_type', 'Shadow price')} |",
            f"| Application | {icp.get('application', 'Capital expenditure decisions')} |",
            f"| Price Trajectory | {icp.get('trajectory', 'Increasing annually')} |",
            f"| Coverage | {icp.get('coverage', 'All major investment decisions')} |",
        ]
        return "\n".join(lines)

    def _md_tcfd_alignment(self, data: Dict[str, Any]) -> str:
        alignment = data.get("tcfd_alignment", {})
        lines = [
            "## 9. TCFD Recommendation Alignment\n",
            "| # | Recommendation | Pillar | Status |",
            "|---|---------------|--------|--------|",
        ]
        passed = 0
        for i, rec in enumerate(TCFD_METRIC_RECS, 1):
            r = alignment.get(rec["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            if status == "pass":
                passed += 1
            lines.append(f"| {i} | {rec['rec']} | {rec['pillar']} | **{icon}** |")
        score = (passed / max(1, len(TCFD_METRIC_RECS))) * 100
        lines.append(f"\n**TCFD Alignment Score:** {_dec(score, 1)}% ({passed}/{len(TCFD_METRIC_RECS)})")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 10. XBRL Tagging Summary\n",
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
            "## 11. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*TCFD Metrics & Targets pillar disclosure.*"

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
            f".pass{{color:{_SUCCESS};font-weight:700;}}.fail{{color:{_DANGER};font-weight:700;}}.pending{{color:{_WARN};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>TCFD Metrics & Targets</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_year","")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        em = data.get("emissions", {})
        total = sum(float(v) for v in em.values())
        return (
            f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(total, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(em.get("scope1",0), 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(em.get("scope2",0), 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Scope 3</div><div class="card-value">{_dec_comma(em.get("scope3",0), 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_ghg_emissions(self, data: Dict[str, Any]) -> str:
        em = data.get("emissions", {})
        rows = ""
        for s in ["scope1", "scope2", "scope3"]:
            rows += f'<tr><td>{s.replace("scope","Scope ").title()}</td><td>{_dec_comma(em.get(s,0), 0)}</td></tr>\n'
        return f'<h2>2. GHG Emissions</h2>\n<table>\n<tr><th>Scope</th><th>Emissions (tCO2e)</th></tr>\n{rows}</table>'

    def _html_transition_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("transition_risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("risk","")}</td><td>{r.get("type","")}</td><td>{r.get("financial_impact","")}</td></tr>\n'
        return f'<h2>3. Transition Risks</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Type</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        rows = ""
        for i, o in enumerate(opps, 1):
            rows += f'<tr><td>{i}</td><td>{o.get("opportunity","")}</td><td>{o.get("type","")}</td><td>{o.get("financial_impact","")}</td></tr>\n'
        return f'<h2>4. Opportunities</h2>\n<table>\n<tr><th>#</th><th>Opportunity</th><th>Type</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_forward_metrics(self, data: Dict[str, Any]) -> str:
        projections = data.get("forward_metrics", {}).get("projections", [])
        rows = ""
        for p in projections:
            rows += f'<tr><td>{p.get("year","")}</td><td>{_dec_comma(p.get("projected",0), 0)}</td><td>{_dec_comma(p.get("target",0), 0)}</td><td>{p.get("scenario","")}</td></tr>\n'
        return f'<h2>5. Forward Metrics</h2>\n<table>\n<tr><th>Year</th><th>Projected</th><th>Target</th><th>Scenario</th></tr>\n{rows}</table>'

    def _html_scenario_analysis(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenario_analysis", {}).get("scenarios", [])
        rows = ""
        for s in scenarios:
            rows += f'<tr><td>{s.get("name","")}</td><td>{s.get("temperature","")}</td><td>{s.get("emissions_impact","")}</td><td>{s.get("financial_impact","")}</td></tr>\n'
        return f'<h2>6. Scenarios</h2>\n<table>\n<tr><th>Scenario</th><th>Temp</th><th>Emissions</th><th>Financial</th></tr>\n{rows}</table>'

    def _html_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("carbon_pricing", {})
        return (
            f'<h2>7. Carbon Pricing</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Current Exposure</td><td>${_dec(cp.get("current_exposure",0))}/tCO2e</td></tr>\n'
            f'<tr><td>Projected 2030</td><td>${_dec(cp.get("projected_2030",0))}/tCO2e</td></tr>\n'
            f'</table>'
        )

    def _html_icp_application(self, data: Dict[str, Any]) -> str:
        icp = data.get("carbon_pricing", {})
        return (
            f'<h2>8. Internal Carbon Price</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Price</td><td>${_dec(icp.get("internal_price",0))}/tCO2e</td></tr>\n'
            f'<tr><td>Type</td><td>{icp.get("price_type","Shadow price")}</td></tr>\n'
            f'</table>'
        )

    def _html_tcfd_alignment(self, data: Dict[str, Any]) -> str:
        alignment = data.get("tcfd_alignment", {})
        rows = ""
        for i, rec in enumerate(TCFD_METRIC_RECS, 1):
            r = alignment.get(rec["id"], {})
            s = r.get("status", "pending")
            cls = "pass" if s == "pass" else ("fail" if s == "fail" else "pending")
            rows += f'<tr><td>{i}</td><td>{rec["rec"]}</td><td>{rec["pillar"]}</td><td class="{cls}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td></tr>\n'
        return f'<h2>9. TCFD Alignment</h2>\n<table>\n<tr><th>#</th><th>Recommendation</th><th>Pillar</th><th>Status</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>10. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>11. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - TCFD metrics</div>'
