# -*- coding: utf-8 -*-
"""
TransitionFinanceReportTemplate - Climate CapEx/OpEx and Taxonomy alignment for PACK-022.

Renders a transition finance report covering climate investment classification,
EU Taxonomy alignment, green bond eligibility, internal carbon pricing, NPV/IRR
investment cases, climate OpEx projections, cost of inaction, and ROI summary.

Sections:
    1. Climate Investment Summary
    2. CapEx Classification (climate vs non-climate)
    3. Category Breakdown
    4. EU Taxonomy Alignment
    5. Green Bond Eligibility
    6. Internal Carbon Pricing Impact
    7. Investment Cases (NPV/IRR)
    8. Climate OpEx Projection
    9. Cost of Inaction Analysis
   10. ROI Summary

Author: GreenLang Team
Version: 22.0.0
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

_MODULE_VERSION = "22.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
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

def _pct_of(part: Any, total: Any) -> Decimal:
    try:
        p = Decimal(str(part))
        t = Decimal(str(total))
        if t == 0:
            return Decimal("0.00")
        return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal("0.00")

class TransitionFinanceReportTemplate:
    """
    Climate CapEx/OpEx and Taxonomy alignment report template.

    Covers climate investment classification, EU Taxonomy alignment,
    green bond eligibility, carbon pricing, investment case analysis,
    and cost of inaction modelling.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_investment_summary(data),
            self._md_capex_classification(data),
            self._md_category_breakdown(data),
            self._md_taxonomy_alignment(data),
            self._md_green_bond(data),
            self._md_carbon_pricing(data),
            self._md_investment_cases(data),
            self._md_opex_projection(data),
            self._md_cost_of_inaction(data),
            self._md_roi_summary(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_investment_summary(data),
            self._html_capex_classification(data),
            self._html_category_breakdown(data),
            self._html_taxonomy_alignment(data),
            self._html_green_bond(data),
            self._html_carbon_pricing(data),
            self._html_investment_cases(data),
            self._html_opex_projection(data),
            self._html_cost_of_inaction(data),
            self._html_roi_summary(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Transition Finance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        capex = data.get("capex", {})
        taxonomy = data.get("taxonomy_alignment", {})
        investments = data.get("investment_cases", [])
        roi = data.get("roi_summary", {})

        result: Dict[str, Any] = {
            "template": "transition_finance_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "investment_summary": {
                "total_climate_capex_eur": str(capex.get("climate_capex_eur", 0)),
                "total_non_climate_capex_eur": str(capex.get("non_climate_capex_eur", 0)),
                "taxonomy_aligned_pct": str(taxonomy.get("aligned_pct", 0)),
                "investment_cases_count": len(investments),
            },
            "capex": capex,
            "categories": data.get("categories", []),
            "taxonomy_alignment": taxonomy,
            "green_bond": data.get("green_bond", {}),
            "carbon_pricing": data.get("carbon_pricing", {}),
            "investment_cases": investments,
            "opex_projection": data.get("opex_projection", []),
            "cost_of_inaction": data.get("cost_of_inaction", {}),
            "roi_summary": roi,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Transition Finance Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_investment_summary(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex", {})
        climate = Decimal(str(capex.get("climate_capex_eur", 0)))
        non_climate = Decimal(str(capex.get("non_climate_capex_eur", 0)))
        total = climate + non_climate
        taxonomy = data.get("taxonomy_alignment", {})
        return (
            "## 1. Climate Investment Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total CapEx | EUR {_dec_comma(total, 0)} |\n"
            f"| Climate CapEx | EUR {_dec_comma(climate, 0)} ({_dec(_pct_of(climate, total))}%) |\n"
            f"| Non-Climate CapEx | EUR {_dec_comma(non_climate, 0)} ({_dec(_pct_of(non_climate, total))}%) |\n"
            f"| EU Taxonomy Aligned | {_dec(taxonomy.get('aligned_pct', 0))}% |\n"
            f"| Green Bond Eligible | EUR {_dec_comma(data.get('green_bond', {}).get('eligible_eur', 0), 0)} |\n"
            f"| Internal Carbon Price | EUR {_dec(data.get('carbon_pricing', {}).get('price_per_tco2e', 0), 0)}/tCO2e |"
        )

    def _md_capex_classification(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex", {})
        climate = Decimal(str(capex.get("climate_capex_eur", 0)))
        non_climate = Decimal(str(capex.get("non_climate_capex_eur", 0)))
        total = climate + non_climate
        return (
            "## 2. CapEx Classification\n\n"
            "| Classification | Amount (EUR) | Share (%) |\n"
            "|---------------|-------------:|:---------:|\n"
            f"| Climate CapEx | {_dec_comma(climate, 0)} | {_dec(_pct_of(climate, total))}% |\n"
            f"| Non-Climate CapEx | {_dec_comma(non_climate, 0)} | {_dec(_pct_of(non_climate, total))}% |\n"
            f"| **Total** | **{_dec_comma(total, 0)}** | **100.00%** |\n\n"
            f"**Climate CapEx Definition:** {capex.get('definition', 'CapEx directly contributing to GHG emission reduction or climate adaptation')}"
        )

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        total = sum(Decimal(str(c.get("amount_eur", 0))) for c in categories)
        lines = [
            "## 3. Category Breakdown\n",
            "| Category | Type | Amount (EUR) | Share (%) | Emissions Impact (tCO2e/yr) |",
            "|----------|------|-------------:|:---------:|----------------------------:|",
        ]
        for cat in categories:
            amt = Decimal(str(cat.get("amount_eur", 0)))
            lines.append(
                f"| {cat.get('name', '-')} | {cat.get('type', '-')} "
                f"| {_dec_comma(amt, 0)} "
                f"| {_dec(_pct_of(amt, total))}% "
                f"| {_dec_comma(cat.get('emissions_impact_tco2e', 0))} |"
            )
        if not categories:
            lines.append("| _No categories_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        tax = data.get("taxonomy_alignment", {})
        activities = tax.get("activities", [])
        lines = [
            "## 4. EU Taxonomy Alignment\n",
            f"**Overall Alignment:** {_dec(tax.get('aligned_pct', 0))}%  \n"
            f"**Eligible Activities:** {len(activities)}  \n"
            f"**DNSH Compliant:** {tax.get('dnsh_compliant', 'N/A')}\n",
            "| Activity | NACE Code | CapEx (EUR) | Aligned | SC Criteria | DNSH | Min Safeguards |",
            "|----------|-----------|------------:|:-------:|:-----------:|:----:|:--------------:|",
        ]
        for act in activities:
            aligned = "Yes" if act.get("aligned", False) else "No"
            sc = "Pass" if act.get("sc_criteria", False) else "Fail"
            dnsh = "Pass" if act.get("dnsh", False) else "Fail"
            ms = "Pass" if act.get("min_safeguards", False) else "Fail"
            lines.append(
                f"| {act.get('name', '-')} | {act.get('nace_code', '-')} "
                f"| {_dec_comma(act.get('capex_eur', 0), 0)} "
                f"| {aligned} | {sc} | {dnsh} | {ms} |"
            )
        if not activities:
            lines.append("| _No taxonomy activities_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_green_bond(self, data: Dict[str, Any]) -> str:
        gb = data.get("green_bond", {})
        projects = gb.get("eligible_projects", [])
        lines = [
            "## 5. Green Bond Eligibility\n",
            f"**Total Eligible:** EUR {_dec_comma(gb.get('eligible_eur', 0), 0)}  \n"
            f"**Framework:** {gb.get('framework', 'ICMA Green Bond Principles')}  \n"
            f"**External Review:** {gb.get('external_review', 'N/A')}\n",
            "| Project | Category | Amount (EUR) | Use of Proceeds | Eligible |",
            "|---------|----------|-------------:|-----------------|:--------:|",
        ]
        for proj in projects:
            eligible = "Yes" if proj.get("eligible", False) else "No"
            lines.append(
                f"| {proj.get('name', '-')} | {proj.get('category', '-')} "
                f"| {_dec_comma(proj.get('amount_eur', 0), 0)} "
                f"| {proj.get('use_of_proceeds', '-')} "
                f"| {eligible} |"
            )
        if not projects:
            lines.append("| _No eligible projects_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("carbon_pricing", {})
        scenarios = cp.get("scenarios", [])
        lines = [
            "## 6. Internal Carbon Pricing Impact\n",
            f"**Current ICP:** EUR {_dec(cp.get('price_per_tco2e', 0), 0)}/tCO2e  \n"
            f"**Methodology:** {cp.get('methodology', 'Shadow pricing')}  \n"
            f"**Applied To:** {cp.get('applied_to', 'All investment decisions > EUR 1M')}\n",
            "| Scenario | Carbon Price (EUR/tCO2e) | Annual Impact (EUR) | NPV Impact (EUR) |",
            "|----------|-------------------------:|--------------------:|------------------:|",
        ]
        for sc in scenarios:
            lines.append(
                f"| {sc.get('name', '-')} "
                f"| {_dec_comma(sc.get('price_per_tco2e', 0), 0)} "
                f"| {_dec_comma(sc.get('annual_impact_eur', 0), 0)} "
                f"| {_dec_comma(sc.get('npv_impact_eur', 0), 0)} |"
            )
        if not scenarios:
            lines.append("| _No scenarios_ | - | - | - |")
        return "\n".join(lines)

    def _md_investment_cases(self, data: Dict[str, Any]) -> str:
        cases = data.get("investment_cases", [])
        lines = [
            "## 7. Investment Cases (NPV/IRR)\n",
            "| # | Project | CapEx (EUR) | NPV (EUR) | IRR (%) | Payback (yrs) | Abatement (tCO2e/yr) | EUR/tCO2e |",
            "|---|---------|------------:|----------:|--------:|--------------:|---------------------:|----------:|",
        ]
        for i, case in enumerate(cases, 1):
            lines.append(
                f"| {i} | {case.get('name', '-')} "
                f"| {_dec_comma(case.get('capex_eur', 0), 0)} "
                f"| {_dec_comma(case.get('npv_eur', 0), 0)} "
                f"| {_dec(case.get('irr_pct', 0), 1)}% "
                f"| {_dec(case.get('payback_years', 0), 1)} "
                f"| {_dec_comma(case.get('abatement_tco2e', 0))} "
                f"| {_dec_comma(case.get('cost_per_tco2e', 0))} |"
            )
        if not cases:
            lines.append("| - | _No investment cases_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_opex_projection(self, data: Dict[str, Any]) -> str:
        projections = data.get("opex_projection", [])
        lines = [
            "## 8. Climate OpEx Projection\n",
            "| Year | Climate OpEx (EUR) | BAU OpEx (EUR) | Net Savings (EUR) | Cumulative Savings (EUR) |",
            "|:----:|-------------------:|---------------:|------------------:|-------------------------:|",
        ]
        for row in projections:
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec_comma(row.get('climate_opex_eur', 0), 0)} "
                f"| {_dec_comma(row.get('bau_opex_eur', 0), 0)} "
                f"| {_dec_comma(row.get('net_savings_eur', 0), 0)} "
                f"| {_dec_comma(row.get('cumulative_savings_eur', 0), 0)} |"
            )
        if not projections:
            lines.append("| - | _No projections_ | - | - | - |")
        return "\n".join(lines)

    def _md_cost_of_inaction(self, data: Dict[str, Any]) -> str:
        coi = data.get("cost_of_inaction", {})
        risks = coi.get("risks", [])
        lines = [
            "## 9. Cost of Inaction Analysis\n",
            f"**Total Estimated Cost of Inaction (10yr):** EUR {_dec_comma(coi.get('total_10yr_eur', 0), 0)}  \n"
            f"**Methodology:** {coi.get('methodology', 'Scenario-based risk quantification')}\n",
            "| Risk Category | Probability (%) | Impact (EUR) | Expected Loss (EUR) | Timeframe |",
            "|---------------|:---------------:|--------------:|--------------------:|-----------|",
        ]
        for risk in risks:
            prob = Decimal(str(risk.get("probability_pct", 0)))
            impact = Decimal(str(risk.get("impact_eur", 0)))
            expected = (prob / Decimal("100") * impact).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            lines.append(
                f"| {risk.get('category', '-')} "
                f"| {_dec(prob)}% "
                f"| {_dec_comma(impact, 0)} "
                f"| {_dec_comma(expected, 0)} "
                f"| {risk.get('timeframe', '-')} |"
            )
        if not risks:
            lines.append("| _No risk data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_roi_summary(self, data: Dict[str, Any]) -> str:
        roi = data.get("roi_summary", {})
        return (
            "## 10. ROI Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Climate Investment | EUR {_dec_comma(roi.get('total_investment_eur', 0), 0)} |\n"
            f"| Total NPV | EUR {_dec_comma(roi.get('total_npv_eur', 0), 0)} |\n"
            f"| Weighted Average IRR | {_dec(roi.get('weighted_irr_pct', 0), 1)}% |\n"
            f"| Weighted Payback | {_dec(roi.get('weighted_payback_years', 0), 1)} years |\n"
            f"| Total Abatement | {_dec_comma(roi.get('total_abatement_tco2e', 0))} tCO2e/yr |\n"
            f"| Avg Cost per tCO2e | EUR {_dec_comma(roi.get('avg_cost_per_tco2e', 0))} |\n"
            f"| Cost of Inaction Avoided | EUR {_dec_comma(roi.get('inaction_avoided_eur', 0), 0)} |\n"
            f"| Net Benefit (10yr) | EUR {_dec_comma(roi.get('net_benefit_10yr_eur', 0), 0)} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Taxonomy alignment per EU Taxonomy Regulation (EU) 2020/852.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".pass{color:#1b5e20;font-weight:700;}"
            ".fail{color:#c62828;font-weight:700;}"
            ".positive{color:#1b5e20;font-weight:600;}"
            ".negative{color:#c62828;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Transition Finance Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_investment_summary(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex", {})
        climate = Decimal(str(capex.get("climate_capex_eur", 0)))
        non_climate = Decimal(str(capex.get("non_climate_capex_eur", 0)))
        total = climate + non_climate
        tax = data.get("taxonomy_alignment", {})
        return (
            f'<h2>1. Climate Investment Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(total, 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Climate CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(climate, 0)}</div>'
            f'<div class="card-unit">{_dec(_pct_of(climate, total))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Taxonomy Aligned</div>'
            f'<div class="card-value">{_dec(tax.get("aligned_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Green Bond Eligible</div>'
            f'<div class="card-value">EUR {_dec_comma(data.get("green_bond", {}).get("eligible_eur", 0), 0)}</div></div>\n'
            f'</div>'
        )

    def _html_capex_classification(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex", {})
        climate = Decimal(str(capex.get("climate_capex_eur", 0)))
        non_climate = Decimal(str(capex.get("non_climate_capex_eur", 0)))
        total = climate + non_climate
        c_pct = float(_pct_of(climate, total))
        return (
            f'<h2>2. CapEx Classification</h2>\n'
            f'<table>\n'
            f'<tr><th>Classification</th><th>Amount (EUR)</th><th>Share</th><th>Visual</th></tr>\n'
            f'<tr><td>Climate CapEx</td><td>{_dec_comma(climate, 0)}</td>'
            f'<td>{_dec(c_pct)}%</td>'
            f'<td><div style="background:#43a047;height:20px;width:{min(c_pct, 100)}%;border-radius:4px;"></div></td></tr>\n'
            f'<tr><td>Non-Climate CapEx</td><td>{_dec_comma(non_climate, 0)}</td>'
            f'<td>{_dec(100-c_pct)}%</td>'
            f'<td><div style="background:#bdbdbd;height:20px;width:{min(100-c_pct, 100)}%;border-radius:4px;"></div></td></tr>\n'
            f'<tr><th>Total</th><th>{_dec_comma(total, 0)}</th><th>100.00%</th><th></th></tr>\n'
            f'</table>'
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        total = sum(Decimal(str(c.get("amount_eur", 0))) for c in categories)
        rows = ""
        for cat in categories:
            amt = Decimal(str(cat.get("amount_eur", 0)))
            rows += (
                f'<tr><td>{cat.get("name", "-")}</td><td>{cat.get("type", "-")}</td>'
                f'<td>{_dec_comma(amt, 0)}</td>'
                f'<td>{_dec(_pct_of(amt, total))}%</td>'
                f'<td>{_dec_comma(cat.get("emissions_impact_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>3. Category Breakdown</h2>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Type</th><th>Amount (EUR)</th>'
            f'<th>Share</th><th>Emissions Impact (tCO2e/yr)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_taxonomy_alignment(self, data: Dict[str, Any]) -> str:
        tax = data.get("taxonomy_alignment", {})
        activities = tax.get("activities", [])
        rows = ""
        for act in activities:
            a_cls = "pass" if act.get("aligned", False) else "fail"
            a_icon = "&#10004;" if act.get("aligned", False) else "&#10008;"
            sc_icon = "&#10004;" if act.get("sc_criteria", False) else "&#10008;"
            d_icon = "&#10004;" if act.get("dnsh", False) else "&#10008;"
            m_icon = "&#10004;" if act.get("min_safeguards", False) else "&#10008;"
            rows += (
                f'<tr><td>{act.get("name", "-")}</td><td>{act.get("nace_code", "-")}</td>'
                f'<td>{_dec_comma(act.get("capex_eur", 0), 0)}</td>'
                f'<td class="{a_cls}">{a_icon}</td>'
                f'<td>{sc_icon}</td><td>{d_icon}</td><td>{m_icon}</td></tr>\n'
            )
        return (
            f'<h2>4. EU Taxonomy Alignment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Aligned</div>'
            f'<div class="card-value">{_dec(tax.get("aligned_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Eligible Activities</div>'
            f'<div class="card-value">{len(activities)}</div></div>\n'
            f'  <div class="card"><div class="card-label">DNSH</div>'
            f'<div class="card-value">{tax.get("dnsh_compliant", "N/A")}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Activity</th><th>NACE</th><th>CapEx (EUR)</th>'
            f'<th>Aligned</th><th>SC</th><th>DNSH</th><th>Min Safeguards</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_green_bond(self, data: Dict[str, Any]) -> str:
        gb = data.get("green_bond", {})
        projects = gb.get("eligible_projects", [])
        rows = ""
        for proj in projects:
            e_cls = "pass" if proj.get("eligible", False) else "fail"
            e_label = "Yes" if proj.get("eligible", False) else "No"
            rows += (
                f'<tr><td>{proj.get("name", "-")}</td><td>{proj.get("category", "-")}</td>'
                f'<td>{_dec_comma(proj.get("amount_eur", 0), 0)}</td>'
                f'<td>{proj.get("use_of_proceeds", "-")}</td>'
                f'<td class="{e_cls}">{e_label}</td></tr>\n'
            )
        return (
            f'<h2>5. Green Bond Eligibility</h2>\n'
            f'<p><strong>Eligible:</strong> EUR {_dec_comma(gb.get("eligible_eur", 0), 0)} | '
            f'<strong>Framework:</strong> {gb.get("framework", "ICMA GBP")}</p>\n'
            f'<table>\n'
            f'<tr><th>Project</th><th>Category</th><th>Amount (EUR)</th>'
            f'<th>Use of Proceeds</th><th>Eligible</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("carbon_pricing", {})
        scenarios = cp.get("scenarios", [])
        rows = ""
        for sc in scenarios:
            rows += (
                f'<tr><td>{sc.get("name", "-")}</td>'
                f'<td>EUR {_dec_comma(sc.get("price_per_tco2e", 0), 0)}</td>'
                f'<td>EUR {_dec_comma(sc.get("annual_impact_eur", 0), 0)}</td>'
                f'<td>EUR {_dec_comma(sc.get("npv_impact_eur", 0), 0)}</td></tr>\n'
            )
        return (
            f'<h2>6. Internal Carbon Pricing Impact</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Current ICP</div>'
            f'<div class="card-value">EUR {_dec(cp.get("price_per_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">per tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Methodology</div>'
            f'<div class="card-value">{cp.get("methodology", "Shadow")}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>Carbon Price</th><th>Annual Impact</th>'
            f'<th>NPV Impact</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_investment_cases(self, data: Dict[str, Any]) -> str:
        cases = data.get("investment_cases", [])
        rows = ""
        for i, case in enumerate(cases, 1):
            npv = float(Decimal(str(case.get("npv_eur", 0))))
            npv_cls = "positive" if npv >= 0 else "negative"
            rows += (
                f'<tr><td>{i}</td><td><strong>{case.get("name", "-")}</strong></td>'
                f'<td>{_dec_comma(case.get("capex_eur", 0), 0)}</td>'
                f'<td class="{npv_cls}">{_dec_comma(case.get("npv_eur", 0), 0)}</td>'
                f'<td>{_dec(case.get("irr_pct", 0), 1)}%</td>'
                f'<td>{_dec(case.get("payback_years", 0), 1)}</td>'
                f'<td>{_dec_comma(case.get("abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(case.get("cost_per_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>7. Investment Cases (NPV/IRR)</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Project</th><th>CapEx (EUR)</th><th>NPV (EUR)</th>'
            f'<th>IRR</th><th>Payback (yrs)</th><th>Abatement (tCO2e/yr)</th>'
            f'<th>EUR/tCO2e</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_opex_projection(self, data: Dict[str, Any]) -> str:
        projections = data.get("opex_projection", [])
        rows = ""
        for row in projections:
            savings = float(Decimal(str(row.get("net_savings_eur", 0))))
            s_cls = "positive" if savings >= 0 else "negative"
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec_comma(row.get("climate_opex_eur", 0), 0)}</td>'
                f'<td>{_dec_comma(row.get("bau_opex_eur", 0), 0)}</td>'
                f'<td class="{s_cls}">{_dec_comma(row.get("net_savings_eur", 0), 0)}</td>'
                f'<td>{_dec_comma(row.get("cumulative_savings_eur", 0), 0)}</td></tr>\n'
            )
        return (
            f'<h2>8. Climate OpEx Projection</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Climate OpEx (EUR)</th><th>BAU OpEx (EUR)</th>'
            f'<th>Net Savings</th><th>Cumulative Savings</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cost_of_inaction(self, data: Dict[str, Any]) -> str:
        coi = data.get("cost_of_inaction", {})
        risks = coi.get("risks", [])
        rows = ""
        for risk in risks:
            prob = Decimal(str(risk.get("probability_pct", 0)))
            impact = Decimal(str(risk.get("impact_eur", 0)))
            expected = (prob / Decimal("100") * impact).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            rows += (
                f'<tr><td>{risk.get("category", "-")}</td>'
                f'<td>{_dec(prob)}%</td>'
                f'<td>{_dec_comma(impact, 0)}</td>'
                f'<td class="negative">{_dec_comma(expected, 0)}</td>'
                f'<td>{risk.get("timeframe", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Cost of Inaction Analysis</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">10-Year Cost of Inaction</div>'
            f'<div class="card-value negative">EUR {_dec_comma(coi.get("total_10yr_eur", 0), 0)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Risk Category</th><th>Probability</th><th>Impact (EUR)</th>'
            f'<th>Expected Loss</th><th>Timeframe</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_roi_summary(self, data: Dict[str, Any]) -> str:
        roi = data.get("roi_summary", {})
        return (
            f'<h2>10. ROI Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Investment</div>'
            f'<div class="card-value">EUR {_dec_comma(roi.get("total_investment_eur", 0), 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Total NPV</div>'
            f'<div class="card-value">EUR {_dec_comma(roi.get("total_npv_eur", 0), 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Weighted IRR</div>'
            f'<div class="card-value">{_dec(roi.get("weighted_irr_pct", 0), 1)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Abatement</div>'
            f'<div class="card-value">{_dec_comma(roi.get("total_abatement_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e/yr</div></div>\n'
            f'  <div class="card"><div class="card-label">Net Benefit (10yr)</div>'
            f'<div class="card-value">EUR {_dec_comma(roi.get("net_benefit_10yr_eur", 0), 0)}</div></div>\n'
            f'</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'EU Taxonomy Regulation (EU) 2020/852.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
