# -*- coding: utf-8 -*-
"""
ScenarioComparisonReportTemplate - Multi-scenario pathway comparison for PACK-028.

Renders a 5-scenario comparison matrix with investment deltas, risk-return
analysis, optimal pathway recommendation, and sensitivity analysis.
Multi-format (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Scenario Definitions (NZE/WB2C/2C/APS/STEPS)
    3.  Scenario Comparison Matrix
    4.  Pathway Trajectories by Scenario
    5.  Investment Requirements per Scenario
    6.  Investment Delta Analysis
    7.  Risk-Return Analysis
    8.  Technology Requirements by Scenario
    9.  Carbon Budget Consumption
    10. Optimal Pathway Recommendation
    11. Sensitivity Analysis
    12. XBRL Tagging Summary
    13. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
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

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "scenario_comparison_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"

SCENARIO_DEFS: List[Dict[str, Any]] = [
    {"id": "nze", "name": "Net Zero Emissions (NZE)", "temp": "1.5C", "prob": "50%", "iea": "IEA NZE 2050", "sbti": "1.5C-aligned", "color": "#1b5e20"},
    {"id": "wb2c", "name": "Well-Below 2C (WB2C)", "temp": "<2C", "prob": "66%", "iea": "IEA WB2C", "sbti": "WB2C-aligned", "color": "#388e3c"},
    {"id": "2c", "name": "2 Degrees (2DS)", "temp": "2C", "prob": "50%", "iea": "IEA 2DS", "sbti": "2C-aligned", "color": "#ef6c00"},
    {"id": "aps", "name": "Announced Pledges (APS)", "temp": "1.7C", "prob": "N/A", "iea": "IEA APS", "sbti": "Announced pledges", "color": "#ffa726"},
    {"id": "steps", "name": "Stated Policies (STEPS)", "temp": "2.4C", "prob": "N/A", "iea": "IEA STEPS", "sbti": "Current policies", "color": "#c62828"},
]

XBRL_SCENARIO_TAGS: Dict[str, str] = {
    "recommended_scenario": "gl:RecommendedScenarioIdentifier",
    "scenario_count": "gl:ScenarioComparisonCount",
    "nze_capex": "gl:NZEScenarioCapExRequirement",
    "investment_delta": "gl:InvestmentDeltaVsBAU",
    "risk_score": "gl:ScenarioRiskScore",
    "carbon_budget_consumed": "gl:CarbonBudgetConsumedPercentage",
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

def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg: ip = ip[1:]
        fmt = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = ch + fmt
        if neg: fmt = "-" + fmt
        if len(parts) > 1: fmt += "." + parts[1]
        return fmt
    except Exception:
        return str(val)

def _scenario_lookup(sid: str) -> Optional[Dict[str, Any]]:
    for s in SCENARIO_DEFS:
        if s["id"] == sid:
            return s
    return None

class ScenarioComparisonReportTemplate:
    """
    Multi-scenario pathway comparison report template.

    Compares 5 IEA/SBTi scenarios (NZE, WB2C, 2DS, APS, STEPS) with
    investment deltas, risk-return analysis, and optimal pathway recommendation.
    Supports MD, HTML, JSON, and PDF.

    Example:
        >>> template = ScenarioComparisonReportTemplate()
        >>> data = {"org_name": "SteelCo", "sector_id": "steel", "scenarios": {...}}
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_exec_summary(data),
            self._md_scenario_defs(data), self._md_comparison_matrix(data),
            self._md_trajectories(data), self._md_investment(data),
            self._md_investment_delta(data), self._md_risk_return(data),
            self._md_tech_requirements(data), self._md_carbon_budget(data),
            self._md_recommendation(data), self._md_sensitivity(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_exec_summary(data),
            self._html_scenario_defs(data), self._html_comparison_matrix(data),
            self._html_trajectories(data), self._html_investment(data),
            self._html_investment_delta(data), self._html_risk_return(data),
            self._html_tech_requirements(data), self._html_carbon_budget(data),
            self._html_recommendation(data), self._html_sensitivity(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Scenario Comparison - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        scenarios = data.get("scenarios", {})
        recommended = data.get("recommended_scenario", "nze")
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "sector_id": data.get("sector_id", ""),
            "scenario_definitions": SCENARIO_DEFS,
            "scenario_data": scenarios,
            "recommended_scenario": recommended,
            "investment_comparison": data.get("investment_comparison", {}),
            "risk_return": data.get("risk_return", {}),
            "carbon_budget": data.get("carbon_budget", {}),
            "sensitivity": data.get("sensitivity", {}),
            "xbrl_tags": {k: XBRL_SCENARIO_TAGS[k] for k in XBRL_SCENARIO_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data),
                "structured_data": self.render_json(data),
                "metadata": {"title": f"Scenario Comparison - {data.get('org_name','')}", "author": "GreenLang PACK-028"}}

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Scenario Comparison Report\n\n**Organization:** {data.get('org_name','')}  \n**Sector:** {data.get('sector_id','').replace('_',' ').title()}  \n**Report Date:** {ts}  \n**Pack:** PACK-028 v{_MODULE_VERSION}\n\n---"

    def _md_exec_summary(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        rec_def = _scenario_lookup(rec) or {}
        scenarios = data.get("scenarios", {})
        inv = data.get("investment_comparison", {})
        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |", f"|-----|-------|",
            f"| Scenarios Compared | {len(SCENARIO_DEFS)} |",
            f"| Recommended Pathway | **{rec_def.get('name', rec)}** ({rec_def.get('temp', '')}) |",
            f"| NZE CapEx (total) | EUR {_dec_comma(inv.get('nze_total', 0))} |",
            f"| STEPS CapEx (total) | EUR {_dec_comma(inv.get('steps_total', 0))} |",
            f"| Investment Delta (NZE vs STEPS) | EUR {_dec_comma(float(inv.get('nze_total', 0)) - float(inv.get('steps_total', 0)))} |",
        ]
        return "\n".join(lines)

    def _md_scenario_defs(self, data: Dict[str, Any]) -> str:
        lines = ["## 2. Scenario Definitions\n",
                  "| # | Scenario | Temp Target | Probability | IEA Reference | SBTi Alignment |",
                  "|---|----------|:-----------:|:-----------:|:-------------|:--------------|"]
        for i, s in enumerate(SCENARIO_DEFS, 1):
            lines.append(f"| {i} | {s['name']} | {s['temp']} | {s['prob']} | {s['iea']} | {s['sbti']} |")
        return "\n".join(lines)

    def _md_comparison_matrix(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        lines = ["## 3. Scenario Comparison Matrix\n",
                  "| Dimension |" + "|".join(f" {s['id'].upper()} " for s in SCENARIO_DEFS) + "|",
                  "|-----------|" + "|".join("------:|" for _ in SCENARIO_DEFS)]
        dims = ["intensity_2030", "intensity_2050", "total_reduction_pct", "annual_rate_pct",
                "capex_total_eur", "carbon_budget_consumed_pct", "risk_score"]
        dim_labels = ["Intensity 2030", "Intensity 2050", "Total Reduction (%)", "Annual Rate (%)",
                      "CapEx (EUR)", "Carbon Budget Used (%)", "Risk Score (1-10)"]
        for dim, label in zip(dims, dim_labels):
            vals = []
            for s in SCENARIO_DEFS:
                sc_data = scenarios.get(s["id"], {})
                v = sc_data.get(dim, "-")
                if isinstance(v, (int, float)):
                    if "eur" in dim.lower():
                        vals.append(f" {_dec_comma(v)} ")
                    else:
                        vals.append(f" {_dec(v, 2)} ")
                else:
                    vals.append(f" {v} ")
            lines.append(f"| {label} |" + "|".join(vals) + "|")
        return "\n".join(lines)

    def _md_trajectories(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        years = [2025, 2030, 2035, 2040, 2045, 2050]
        lines = ["## 4. Pathway Trajectories\n",
                  "| Year |" + "|".join(f" {s['id'].upper()} " for s in SCENARIO_DEFS) + "|",
                  "|------|" + "|".join("------:|" for _ in SCENARIO_DEFS)]
        for yr in years:
            vals = []
            for s in SCENARIO_DEFS:
                trajectory = scenarios.get(s["id"], {}).get("trajectory", {})
                v = trajectory.get(str(yr), trajectory.get(yr, "-"))
                vals.append(f" {_dec(v, 4) if isinstance(v, (int, float)) else v} ")
            lines.append(f"| {yr} |" + "|".join(vals) + "|")
        return "\n".join(lines)

    def _md_investment(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_comparison", {})
        lines = ["## 5. Investment Requirements\n",
                  "| Scenario | CapEx (EUR) | OpEx Delta (EUR/yr) | Total Cost of Ownership | Payback |",
                  "|----------|------------|-------------------:|----------------------:|--------:|"]
        for s in SCENARIO_DEFS:
            sc = inv.get(s["id"], {})
            lines.append(
                f"| {s['name']} | {_dec_comma(sc.get('capex', 0))} | {_dec_comma(sc.get('opex_delta', 0))} "
                f"| {_dec_comma(sc.get('tco', 0))} | {_dec(sc.get('payback_years', 0), 1)} yr |"
            )
        return "\n".join(lines)

    def _md_investment_delta(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_comparison", {})
        steps_capex = float(inv.get("steps", {}).get("capex", 0))
        lines = ["## 6. Investment Delta (vs. STEPS)\n",
                  "| Scenario | CapEx | Delta vs STEPS | Additional % |",
                  "|----------|------:|---------------:|--------------:|"]
        for s in SCENARIO_DEFS:
            capex = float(inv.get(s["id"], {}).get("capex", 0))
            delta = capex - steps_capex
            pct = ((delta / steps_capex) * 100) if steps_capex > 0 else 0
            lines.append(f"| {s['name']} | {_dec_comma(capex)} | {'+' if delta > 0 else ''}{_dec_comma(delta)} | {'+' if pct > 0 else ''}{_dec(pct)}% |")
        return "\n".join(lines)

    def _md_risk_return(self, data: Dict[str, Any]) -> str:
        rr = data.get("risk_return", {})
        lines = ["## 7. Risk-Return Analysis\n",
                  "| Scenario | Risk Score | Return Score | Risk-Adjusted Return | Regulatory Risk | Stranded Asset Risk |",
                  "|----------|-----------|-------------|---------------------|----------------|-------------------|"]
        for s in SCENARIO_DEFS:
            sc = rr.get(s["id"], {})
            lines.append(
                f"| {s['name']} | {_dec(sc.get('risk', 5), 1)}/10 | {_dec(sc.get('return', 5), 1)}/10 "
                f"| {_dec(sc.get('risk_adjusted', 5), 1)}/10 | {sc.get('regulatory_risk', 'Medium')} "
                f"| {sc.get('stranded_asset_risk', 'Medium')} |"
            )
        return "\n".join(lines)

    def _md_tech_requirements(self, data: Dict[str, Any]) -> str:
        tech_reqs = data.get("tech_by_scenario", {})
        lines = ["## 8. Technology Requirements by Scenario\n"]
        for s in SCENARIO_DEFS:
            techs = tech_reqs.get(s["id"], [])
            if techs:
                lines.append(f"\n### {s['name']}\n")
                lines.append("| Technology | Adoption by 2030 | Adoption by 2050 | TRL |")
                lines.append("|-----------|:----------------:|:----------------:|----:|")
                for t in techs:
                    lines.append(f"| {t.get('name','')} | {_dec(t.get('adoption_2030',0))}% | {_dec(t.get('adoption_2050',0))}% | {t.get('trl',0)} |")
        if not tech_reqs:
            lines.append("_Technology breakdown by scenario not yet available._")
        return "\n".join(lines)

    def _md_carbon_budget(self, data: Dict[str, Any]) -> str:
        cb = data.get("carbon_budget", {})
        lines = ["## 9. Carbon Budget Analysis\n",
                  f"**Global Carbon Budget (1.5C):** {_dec_comma(cb.get('global_budget_gtco2', 0))} GtCO2",
                  f"**Company Allocated Budget:** {_dec_comma(cb.get('company_budget_tco2e', 0))} tCO2e\n",
                  "| Scenario | Cumulative Emissions (tCO2e) | Budget Consumed (%) | Budget Remaining |",
                  "|----------|----------------------------:|--------------------:|-----------------:|"]
        for s in SCENARIO_DEFS:
            sc = cb.get(s["id"], {})
            cum = float(sc.get("cumulative", 0))
            budget = float(cb.get("company_budget_tco2e", 1))
            consumed = (cum / budget * 100) if budget > 0 else 0
            remaining = budget - cum
            lines.append(f"| {s['name']} | {_dec_comma(cum)} | {_dec(consumed)}% | {_dec_comma(remaining)} |")
        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        rec_def = _scenario_lookup(rec) or {}
        rationale = data.get("recommendation_rationale", [])
        lines = [
            "## 10. Optimal Pathway Recommendation\n",
            f"**Recommended Scenario:** {rec_def.get('name', rec)}  \n"
            f"**Temperature Alignment:** {rec_def.get('temp', '')}  \n"
            f"**SBTi Compatibility:** {rec_def.get('sbti', '')}\n",
        ]
        if rationale:
            lines.append("### Rationale\n")
            for i, r in enumerate(rationale, 1):
                lines.append(f"{i}. {r}")
        else:
            lines.append(
                "### Decision Criteria\n\n"
                "1. **Ambition:** Alignment with 1.5C pathway and SBTi requirements\n"
                "2. **Feasibility:** Technology readiness and implementation timeline\n"
                "3. **Cost-effectiveness:** Risk-adjusted return on investment\n"
                "4. **Regulatory alignment:** Future-proofing against tightening regulations\n"
                "5. **Stakeholder expectations:** Investor and customer expectations"
            )
        return "\n".join(lines)

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity", {})
        params = sensitivity.get("parameters", [])
        lines = ["## 11. Sensitivity Analysis\n"]
        if params:
            lines.append("| Parameter | Low | Base | High | Impact on Recommendation |")
            lines.append("|-----------|-----|------|------|--------------------------|")
            for p in params:
                lines.append(f"| {p.get('name','')} | {p.get('low','')} | {p.get('base','')} | {p.get('high','')} | {p.get('impact','')} |")
        else:
            lines.append(
                "| Carbon Price | EUR 50/tCO2e | EUR 100/tCO2e | EUR 200/tCO2e | Higher prices favour NZE |\n"
                "| Technology Cost | +20% | Base | -30% | Lower costs favour early adoption |\n"
                "| Demand Growth | -2% CAGR | +1% CAGR | +3% CAGR | Higher demand increases urgency |\n"
                "| Discount Rate | 5% | 8% | 12% | Higher rates favour delayed action |"
            )
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        lines = ["## 12. XBRL Tagging Summary\n",
                  "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
                  f"| Recommended Scenario | {XBRL_SCENARIO_TAGS['recommended_scenario']} | {rec} |",
                  f"| Scenario Count | {XBRL_SCENARIO_TAGS['scenario_count']} | {len(SCENARIO_DEFS)} |"]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f"## 13. Audit Trail\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n*5-scenario pathway comparison with risk-return analysis.*"

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
            f".recommended{{background:#c8e6c9;font-weight:700;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Scenario Comparison Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name","")} | <strong>Sector:</strong> {data.get("sector_id","").replace("_"," ").title()} | <strong>Generated:</strong> {ts}</p>'

    def _html_exec_summary(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        rec_def = _scenario_lookup(rec) or {}
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Recommended</div><div class="card-value">{rec_def.get("name", rec)}</div><div class="card-unit">{rec_def.get("temp","")}</div></div>\n'
            f'<div class="card"><div class="card-label">Scenarios</div><div class="card-value">{len(SCENARIO_DEFS)}</div></div>\n'
            f'</div>'
        )

    def _html_scenario_defs(self, data: Dict[str, Any]) -> str:
        rows = "".join(f'<tr><td>{i}</td><td>{s["name"]}</td><td>{s["temp"]}</td><td>{s["prob"]}</td><td>{s["iea"]}</td><td>{s["sbti"]}</td></tr>\n' for i, s in enumerate(SCENARIO_DEFS, 1))
        return f'<h2>2. Scenario Definitions</h2>\n<table>\n<tr><th>#</th><th>Scenario</th><th>Temp</th><th>Prob</th><th>IEA</th><th>SBTi</th></tr>\n{rows}</table>'

    def _html_comparison_matrix(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        header = "<th>Dimension</th>" + "".join(f"<th>{s['id'].upper()}</th>" for s in SCENARIO_DEFS)
        dims = [("Intensity 2030", "intensity_2030"), ("Intensity 2050", "intensity_2050"),
                ("Reduction (%)", "total_reduction_pct"), ("Annual Rate", "annual_rate_pct"),
                ("CapEx", "capex_total_eur"), ("Budget Used", "carbon_budget_consumed_pct")]
        rows = ""
        for label, key in dims:
            cells = ""
            for s in SCENARIO_DEFS:
                v = scenarios.get(s["id"], {}).get(key, "-")
                cells += f"<td>{_dec(v, 2) if isinstance(v, (int, float)) else v}</td>"
            rows += f"<tr><td>{label}</td>{cells}</tr>\n"
        return f'<h2>3. Comparison Matrix</h2>\n<table>\n<tr>{header}</tr>\n{rows}</table>'

    def _html_trajectories(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        header = "<th>Year</th>" + "".join(f"<th>{s['id'].upper()}</th>" for s in SCENARIO_DEFS)
        rows = ""
        for yr in [2025, 2030, 2035, 2040, 2045, 2050]:
            cells = ""
            for s in SCENARIO_DEFS:
                t = scenarios.get(s["id"], {}).get("trajectory", {})
                v = t.get(str(yr), t.get(yr, "-"))
                cells += f"<td>{_dec(v, 4) if isinstance(v, (int, float)) else v}</td>"
            rows += f"<tr><td>{yr}</td>{cells}</tr>\n"
        return f'<h2>4. Trajectories</h2>\n<table>\n<tr>{header}</tr>\n{rows}</table>'

    def _html_investment(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_comparison", {})
        rec = data.get("recommended_scenario", "nze")
        rows = ""
        for s in SCENARIO_DEFS:
            sc = inv.get(s["id"], {})
            cls = ' class="recommended"' if s["id"] == rec else ""
            rows += f'<tr{cls}><td>{s["name"]}</td><td>EUR {_dec_comma(sc.get("capex",0))}</td><td>EUR {_dec_comma(sc.get("opex_delta",0))}</td><td>{_dec(sc.get("payback_years",0),1)} yr</td></tr>\n'
        return f'<h2>5. Investment</h2>\n<table>\n<tr><th>Scenario</th><th>CapEx</th><th>OpEx Delta</th><th>Payback</th></tr>\n{rows}</table>'

    def _html_investment_delta(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_comparison", {})
        steps = float(inv.get("steps", {}).get("capex", 0))
        rows = ""
        for s in SCENARIO_DEFS:
            capex = float(inv.get(s["id"], {}).get("capex", 0))
            delta = capex - steps
            rows += f'<tr><td>{s["name"]}</td><td>EUR {_dec_comma(capex)}</td><td>{"+" if delta > 0 else ""}{_dec_comma(delta)}</td></tr>\n'
        return f'<h2>6. Investment Delta</h2>\n<table>\n<tr><th>Scenario</th><th>CapEx</th><th>Delta vs STEPS</th></tr>\n{rows}</table>'

    def _html_risk_return(self, data: Dict[str, Any]) -> str:
        rr = data.get("risk_return", {})
        rows = ""
        for s in SCENARIO_DEFS:
            sc = rr.get(s["id"], {})
            rows += f'<tr><td>{s["name"]}</td><td>{_dec(sc.get("risk",5),1)}/10</td><td>{_dec(sc.get("return",5),1)}/10</td><td>{_dec(sc.get("risk_adjusted",5),1)}/10</td></tr>\n'
        return f'<h2>7. Risk-Return</h2>\n<table>\n<tr><th>Scenario</th><th>Risk</th><th>Return</th><th>Adjusted</th></tr>\n{rows}</table>'

    def _html_tech_requirements(self, data: Dict[str, Any]) -> str:
        return f'<h2>8. Technology Requirements</h2>\n<p><em>See Markdown output for detailed per-scenario technology breakdown.</em></p>'

    def _html_carbon_budget(self, data: Dict[str, Any]) -> str:
        cb = data.get("carbon_budget", {})
        rows = ""
        for s in SCENARIO_DEFS:
            sc = cb.get(s["id"], {})
            cum = float(sc.get("cumulative", 0))
            budget = float(cb.get("company_budget_tco2e", 1))
            consumed = (cum / budget * 100) if budget > 0 else 0
            rows += f'<tr><td>{s["name"]}</td><td>{_dec_comma(cum)}</td><td>{_dec(consumed)}%</td></tr>\n'
        return f'<h2>9. Carbon Budget</h2>\n<table>\n<tr><th>Scenario</th><th>Cumulative</th><th>Budget Used</th></tr>\n{rows}</table>'

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        rec_def = _scenario_lookup(rec) or {}
        return f'<h2>10. Recommendation</h2>\n<div class="summary-cards"><div class="card recommended"><div class="card-label">Recommended</div><div class="card-value">{rec_def.get("name", rec)}</div><div class="card-unit">{rec_def.get("temp","")} - {rec_def.get("sbti","")}</div></div></div>'

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        return f'<h2>11. Sensitivity</h2>\n<p><em>See Markdown output for sensitivity parameter analysis.</em></p>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_scenario", "nze")
        return f'<h2>12. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n<tr><td>Recommended</td><td><code>{XBRL_SCENARIO_TAGS["recommended_scenario"]}</code></td><td>{rec}</td></tr>\n<tr><td>Count</td><td><code>{XBRL_SCENARIO_TAGS["scenario_count"]}</code></td><td>{len(SCENARIO_DEFS)}</td></tr>\n</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>13. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-028 on {ts} - 5-scenario comparison</div>'
