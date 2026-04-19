# -*- coding: utf-8 -*-
"""
ActionPlanDocumentTemplate - Race to Zero action plan for PACK-025.

Renders the comprehensive Race to Zero action plan document following the
10-section structure from the Race to Zero Interpretation Guide. Includes
Gantt chart visualization, budget allocation tables, risk mitigation
strategies, and detailed decarbonization roadmaps.

Sections:
    1. Executive Summary
    2. Governance & Oversight
    3. Baseline Emissions
    4. Science-Based Targets
    5. Decarbonization Roadmap
    6. Scope 3 Engagement Strategy
    7. Offset & Removal Strategy
    8. Climate Finance Plan
    9. Reporting & Verification
    10. Just Transition Plan
    + Gantt Chart, Budget Allocation, Risk Mitigation

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
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

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "action_plan_document"

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

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0

class ActionPlanDocumentTemplate:
    """Race to Zero action plan document template for PACK-025.

    Generates the comprehensive 10-section action plan following the Race
    to Zero Interpretation Guide structure with Gantt chart visualization,
    budget allocation tables, and risk mitigation strategies.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    # Default decarbonization levers
    DEFAULT_LEVERS = [
        {"lever": "Energy Efficiency", "scope": "S1+S2", "potential_pct": 15,
         "timeline": "2025-2028", "capex_usd": 500000},
        {"lever": "Renewable Electricity", "scope": "S2", "potential_pct": 25,
         "timeline": "2025-2030", "capex_usd": 1200000},
        {"lever": "Electrification of Heat", "scope": "S1", "potential_pct": 10,
         "timeline": "2026-2032", "capex_usd": 800000},
        {"lever": "Fleet Electrification", "scope": "S1", "potential_pct": 8,
         "timeline": "2026-2035", "capex_usd": 600000},
        {"lever": "Process Optimization", "scope": "S1", "potential_pct": 5,
         "timeline": "2025-2027", "capex_usd": 200000},
        {"lever": "Supplier Engagement", "scope": "S3", "potential_pct": 20,
         "timeline": "2025-2035", "capex_usd": 150000},
        {"lever": "Product Design", "scope": "S3", "potential_pct": 10,
         "timeline": "2026-2035", "capex_usd": 400000},
        {"lever": "Logistics Optimization", "scope": "S3", "potential_pct": 5,
         "timeline": "2025-2030", "capex_usd": 300000},
    ]

    # Default risk categories
    DEFAULT_RISKS = [
        {"risk": "Technology availability", "category": "Technical", "likelihood": "Medium",
         "impact": "High", "mitigation": "Diversified technology portfolio; pilot programs"},
        {"risk": "Regulatory changes", "category": "Regulatory", "likelihood": "Medium",
         "impact": "Medium", "mitigation": "Regulatory monitoring; flexible compliance strategy"},
        {"risk": "Cost escalation", "category": "Financial", "likelihood": "High",
         "impact": "Medium", "mitigation": "Phased investment; contingency budget (15%)"},
        {"risk": "Supply chain disruption", "category": "Operational", "likelihood": "Medium",
         "impact": "High", "mitigation": "Multi-sourcing; supplier resilience assessment"},
        {"risk": "Talent shortage", "category": "Human Capital", "likelihood": "Medium",
         "impact": "Medium", "mitigation": "Training programs; external partnerships"},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the action plan document as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_governance(data),
            self._md_baseline(data),
            self._md_science_targets(data),
            self._md_decarbonization_roadmap(data),
            self._md_scope3_engagement(data),
            self._md_offset_removal(data),
            self._md_climate_finance(data),
            self._md_reporting_verification(data),
            self._md_just_transition(data),
            self._md_gantt_chart(data),
            self._md_budget_allocation(data),
            self._md_risk_mitigation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the action plan document as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_governance(data),
            self._html_baseline(data),
            self._html_targets(data),
            self._html_roadmap(data),
            self._html_scope3(data),
            self._html_offset(data),
            self._html_finance(data),
            self._html_reporting(data),
            self._html_just_transition(data),
            self._html_gantt(data),
            self._html_budget(data),
            self._html_risks(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Action Plan</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the action plan document as structured JSON."""
        self.generated_at = utcnow()
        baseline = data.get("baseline", {})
        targets = data.get("targets", {})
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        budget = data.get("budget", {})

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector": data.get("sector", ""),
            "baseline": baseline,
            "targets": {
                "interim": data.get("interim_target", {}),
                "longterm": data.get("longterm_target", {}),
            },
            "governance": data.get("governance", {}),
            "decarbonization_roadmap": {
                "levers": levers,
                "total_reduction_potential_pct": sum(l.get("potential_pct", 0) for l in levers),
                "total_capex_usd": sum(l.get("capex_usd", 0) for l in levers),
            },
            "scope3_engagement": data.get("scope3_engagement", {}),
            "offset_strategy": data.get("offset_strategy", {}),
            "climate_finance": budget,
            "reporting": data.get("reporting", {}),
            "just_transition": data.get("just_transition", {}),
            "milestones": data.get("milestones", []),
            "risks": data.get("risks", self.DEFAULT_RISKS),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Executive Summary
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        sheets["Executive Summary"] = [
            {"Field": "Organization", "Value": data.get("org_name", "")},
            {"Field": "Sector", "Value": data.get("sector", "")},
            {"Field": "Baseline Year", "Value": baseline.get("year", "")},
            {"Field": "Baseline Emissions (tCO2e)", "Value": baseline.get("total_tco2e", 0)},
            {"Field": "Interim Target Year", "Value": interim.get("year", 2030)},
            {"Field": "Interim Reduction (%)", "Value": interim.get("reduction_pct", 50)},
            {"Field": "Net-Zero Target Year", "Value": longterm.get("year", 2050)},
            {"Field": "Min Reduction (%)", "Value": longterm.get("min_reduction_pct", 90)},
        ]

        # Sheet 2: Governance
        gov = data.get("governance", {})
        roles = gov.get("roles", [])
        gov_rows: List[Dict[str, Any]] = []
        for role in roles:
            gov_rows.append({
                "Role": role.get("role", ""),
                "Name": role.get("name", ""),
                "Responsibility": role.get("responsibility", ""),
                "Reporting": role.get("reporting_to", ""),
            })
        sheets["Governance"] = gov_rows

        # Sheet 3: Decarbonization Roadmap
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        lever_rows: List[Dict[str, Any]] = []
        for lever in levers:
            lever_rows.append({
                "Lever": lever.get("lever", ""),
                "Scope": lever.get("scope", ""),
                "Reduction Potential (%)": lever.get("potential_pct", 0),
                "Timeline": lever.get("timeline", ""),
                "CAPEX (USD)": lever.get("capex_usd", 0),
                "Status": lever.get("status", "Planned"),
            })
        sheets["Decarbonization Roadmap"] = lever_rows

        # Sheet 4: Milestones / Gantt Data
        milestones = data.get("milestones", [])
        ms_rows: List[Dict[str, Any]] = []
        for ms in milestones:
            ms_rows.append({
                "Phase": ms.get("phase", ""),
                "Task": ms.get("task", ms.get("milestone", "")),
                "Start": ms.get("start", ""),
                "End": ms.get("end", ""),
                "Owner": ms.get("owner", ""),
                "Dependencies": ms.get("dependencies", ""),
                "Status": ms.get("status", "Planned"),
            })
        sheets["Milestones"] = ms_rows

        # Sheet 5: Budget Allocation
        budget_items = data.get("budget", {}).get("items", [])
        budget_rows: List[Dict[str, Any]] = []
        for item in budget_items:
            budget_rows.append({
                "Category": item.get("category", ""),
                "Description": item.get("description", ""),
                "Year 1 (USD)": item.get("year1_usd", 0),
                "Year 2 (USD)": item.get("year2_usd", 0),
                "Year 3 (USD)": item.get("year3_usd", 0),
                "Year 4-5 (USD)": item.get("year4_5_usd", 0),
                "Total (USD)": item.get("total_usd", 0),
            })
        sheets["Budget Allocation"] = budget_rows

        # Sheet 6: Risk Register
        risks = data.get("risks", self.DEFAULT_RISKS)
        risk_rows: List[Dict[str, Any]] = []
        for risk in risks:
            risk_rows.append({
                "Risk": risk.get("risk", ""),
                "Category": risk.get("category", ""),
                "Likelihood": risk.get("likelihood", ""),
                "Impact": risk.get("impact", ""),
                "Risk Level": risk.get("risk_level", ""),
                "Mitigation": risk.get("mitigation", ""),
                "Owner": risk.get("owner", ""),
            })
        sheets["Risk Register"] = risk_rows

        # Sheet 7: Scope 3 Engagement
        s3_strategies = data.get("scope3_engagement", {}).get("strategies", [])
        s3_rows: List[Dict[str, Any]] = []
        for s in s3_strategies:
            s3_rows.append({
                "Category": s.get("category", ""),
                "Strategy": s.get("strategy", ""),
                "Target Suppliers": s.get("target_suppliers", ""),
                "Coverage (%)": s.get("coverage_pct", ""),
                "Timeline": s.get("timeline", ""),
            })
        sheets["Scope 3 Engagement"] = s3_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Action Plan\n\n"
            f"**Organization:** {org}  \n"
            f"**Sector:** {data.get('sector', '')}  \n"
            f"**Plan Period:** {data.get('plan_period', '2025-2050')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Document ID:** {_new_uuid()}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        total_capex = sum(l.get("capex_usd", 0) for l in levers)
        total_potential = sum(l.get("potential_pct", 0) for l in levers)

        return (
            f"## 1. Executive Summary\n\n"
            f"This action plan outlines {org}'s comprehensive strategy to achieve "
            f"net-zero greenhouse gas emissions by {longterm.get('year', 2050)}, "
            f"in alignment with the Race to Zero campaign requirements and science-based "
            f"1.5C pathways.\n\n"
            f"### Key Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Emissions | {_dec_comma(baseline.get('total_tco2e', 0))} tCO2e ({baseline.get('year', 'N/A')}) |\n"
            f"| Interim Target | {_pct(interim.get('reduction_pct', 50))} reduction by {interim.get('year', 2030)} |\n"
            f"| Net-Zero Target | {longterm.get('year', 2050)} |\n"
            f"| Decarbonization Levers | {len(levers)} identified |\n"
            f"| Total Reduction Potential | {_pct(total_potential)} |\n"
            f"| Estimated Investment | ${_dec_comma(total_capex)} |\n"
            f"| Scope Coverage | S1 + S2 + S3 |\n"
            f"| Pathway Alignment | {interim.get('pathway', '1.5C no/limited overshoot')} |"
        )

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        roles = gov.get("roles", [])

        lines = [
            "## 2. Governance & Oversight\n",
            f"**Board Oversight:** {gov.get('board_oversight', 'Board-level climate committee')}  \n"
            f"**Executive Sponsor:** {gov.get('executive_sponsor', 'Chief Sustainability Officer')}  \n"
            f"**Review Frequency:** {gov.get('review_frequency', 'Quarterly')}\n",
        ]

        if roles:
            lines.extend([
                "### Key Roles and Responsibilities\n",
                "| Role | Name | Responsibility | Reports To |",
                "|------|------|----------------|-----------|",
            ])
            for role in roles:
                lines.append(
                    f"| {role.get('role', '-')} | {role.get('name', '-')} "
                    f"| {role.get('responsibility', '-')} | {role.get('reporting_to', '-')} |"
                )
        else:
            lines.extend([
                "### Recommended Governance Structure\n",
                "| Level | Body | Frequency | Responsibility |",
                "|-------|------|:---------:|----------------|",
                "| Board | Climate Committee | Quarterly | Strategic oversight and target approval |",
                "| Executive | CSO / Sustainability Director | Monthly | Plan execution and resource allocation |",
                "| Operational | Climate Action Working Group | Bi-weekly | Project implementation and tracking |",
                "| External | Advisory Panel | Semi-annual | Expert review and recommendations |",
            ])

        policies = gov.get("policies", [])
        if policies:
            lines.append("\n### Supporting Policies\n")
            for p in policies:
                lines.append(f"- **{p.get('name', '')}**: {p.get('description', '')}")

        return "\n".join(lines)

    def _md_baseline(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        total = b.get("total_tco2e", 0)
        s1 = b.get("scope1_tco2e", 0)
        s2 = b.get("scope2_tco2e", 0)
        s3 = b.get("scope3_tco2e", 0)

        lines = [
            "## 3. Baseline Emissions\n",
            f"**Base Year:** {b.get('year', 'N/A')}  \n"
            f"**Standard:** {b.get('methodology', 'GHG Protocol Corporate Standard')}  \n"
            f"**Boundary:** {b.get('boundary', 'Operational control')}\n",
            "| Scope | Emissions (tCO2e) | % of Total | Key Sources |",
            "|-------|------------------:|:----------:|-------------|",
            f"| Scope 1 | {_dec_comma(s1)} | {_pct(_safe_div(s1, max(total, 1)) * 100)} | {b.get('scope1_sources', 'Stationary combustion, mobile, process')} |",
            f"| Scope 2 | {_dec_comma(s2)} | {_pct(_safe_div(s2, max(total, 1)) * 100)} | {b.get('scope2_sources', 'Purchased electricity, heat')} |",
            f"| Scope 3 | {_dec_comma(s3)} | {_pct(_safe_div(s3, max(total, 1)) * 100)} | {b.get('scope3_sources', 'Purchased goods, transport, use of sold products')} |",
            f"| **Total** | **{_dec_comma(total)}** | **100%** | |",
        ]

        # Scope 3 categories if available
        s3_cats = b.get("scope3_categories", [])
        if s3_cats:
            lines.extend([
                "\n### Scope 3 Category Breakdown\n",
                "| Cat | Category | tCO2e | % of S3 | Method |",
                "|:---:|----------|------:|:-------:|--------|",
            ])
            for cat in s3_cats:
                lines.append(
                    f"| {cat.get('id', '-')} | {cat.get('name', '-')} "
                    f"| {_dec_comma(cat.get('tco2e', 0))} "
                    f"| {_pct(cat.get('pct_of_s3', 0))} "
                    f"| {cat.get('method', '-')} |"
                )

        return "\n".join(lines)

    def _md_science_targets(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        baseline = data.get("baseline", {})
        total = baseline.get("total_tco2e", 0)

        interim_target_emissions = total * (1 - interim.get("reduction_pct", 50) / 100.0)
        lt_target_emissions = total * (1 - longterm.get("min_reduction_pct", 90) / 100.0)

        lines = [
            "## 4. Science-Based Targets\n",
            "### Target Summary\n",
            "| Parameter | Interim Target | Long-Term Target |",
            "|-----------|:--------------:|:----------------:|",
            f"| Year | {interim.get('year', 2030)} | {longterm.get('year', 2050)} |",
            f"| Reduction | {_pct(interim.get('reduction_pct', 50))} | {_pct(longterm.get('min_reduction_pct', 90))} |",
            f"| Target Emissions | {_dec_comma(interim_target_emissions)} tCO2e | {_dec_comma(lt_target_emissions)} tCO2e |",
            f"| Scope | {interim.get('scope_coverage', 'S1+S2')} | {longterm.get('scope_coverage', 'S1+S2+S3')} |",
            f"| Pathway | {interim.get('pathway', '1.5C aligned')} | {longterm.get('pathway', 'Net-zero')} |",
            f"| Validation | {interim.get('validation', 'SBTi')} | {longterm.get('validation', 'SBTi Net-Zero')} |",
            f"| Offsets | {interim.get('offset_policy', 'Not permitted')} | {longterm.get('offset_policy', 'Residual only')} |",
        ]

        # Target pathway trajectory
        trajectory = data.get("target_trajectory", [])
        if trajectory:
            lines.extend([
                "\n### Target Pathway Trajectory\n",
                "| Year | Target (tCO2e) | Reduction (%) | Cumulative Reduction |",
                "|:----:|---------------:|:-------------:|:--------------------:|",
            ])
            for t in trajectory:
                lines.append(
                    f"| {t.get('year', '-')} | {_dec_comma(t.get('target_tco2e', 0))} "
                    f"| {_pct(t.get('reduction_pct', 0))} "
                    f"| {_pct(t.get('cumulative_pct', 0))} |"
                )

        return "\n".join(lines)

    def _md_decarbonization_roadmap(self, data: Dict[str, Any]) -> str:
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        total_potential = sum(l.get("potential_pct", 0) for l in levers)
        total_capex = sum(l.get("capex_usd", 0) for l in levers)

        lines = [
            "## 5. Decarbonization Roadmap\n",
            f"**Total Reduction Potential:** {_pct(total_potential)}  \n"
            f"**Total Investment:** ${_dec_comma(total_capex)}\n",
            "### Abatement Levers\n",
            "| # | Lever | Scope | Reduction (%) | Timeline | CAPEX (USD) | Status |",
            "|---|-------|-------|:-------------:|:--------:|------------:|:------:|",
        ]

        for i, lever in enumerate(levers, 1):
            lines.append(
                f"| {i} | {lever.get('lever', '-')} "
                f"| {lever.get('scope', '-')} "
                f"| {_pct(lever.get('potential_pct', 0))} "
                f"| {lever.get('timeline', '-')} "
                f"| ${_dec_comma(lever.get('capex_usd', 0))} "
                f"| {lever.get('status', 'Planned')} |"
            )

        lines.append(
            f"| | **Total** | | **{_pct(total_potential)}** | | **${_dec_comma(total_capex)}** | |"
        )

        # Phase breakdown
        phases = data.get("roadmap_phases", [])
        if phases:
            lines.extend([
                "\n### Phase Breakdown\n",
                "| Phase | Period | Focus | Key Actions | Target Reduction |",
                "|-------|:------:|-------|-------------|:----------------:|",
            ])
            for phase in phases:
                lines.append(
                    f"| {phase.get('name', '-')} | {phase.get('period', '-')} "
                    f"| {phase.get('focus', '-')} | {phase.get('actions', '-')} "
                    f"| {_pct(phase.get('target_reduction_pct', 0))} |"
                )

        return "\n".join(lines)

    def _md_scope3_engagement(self, data: Dict[str, Any]) -> str:
        s3 = data.get("scope3_engagement", {})
        strategies = s3.get("strategies", [])

        lines = [
            "## 6. Scope 3 Engagement Strategy\n",
            f"**Total S3 Emissions:** {_dec_comma(s3.get('total_s3_tco2e', 0))} tCO2e  \n"
            f"**Coverage Target:** {_pct(s3.get('coverage_target_pct', 67))}  \n"
            f"**Engagement Approach:** {s3.get('approach', 'Tiered supplier engagement')}\n",
        ]

        if strategies:
            lines.extend([
                "### Engagement Strategies\n",
                "| # | S3 Category | Strategy | Target Coverage | Timeline | Budget |",
                "|---|-------------|----------|:--------------:|:--------:|-------:|",
            ])
            for i, strat in enumerate(strategies, 1):
                lines.append(
                    f"| {i} | {strat.get('category', '-')} "
                    f"| {strat.get('strategy', '-')} "
                    f"| {_pct(strat.get('coverage_pct', 0))} "
                    f"| {strat.get('timeline', '-')} "
                    f"| ${_dec_comma(strat.get('budget_usd', 0))} |"
                )
        else:
            lines.extend([
                "### Default Engagement Framework\n",
                "| Tier | Coverage | Strategy | Actions |",
                "|:----:|:--------:|----------|---------|",
                "| Tier 1 | Top 20 suppliers (80% of S3) | Direct engagement | Joint reduction targets, technical support |",
                "| Tier 2 | Next 50 suppliers (15% of S3) | CDP Supply Chain | Disclosure request, benchmark sharing |",
                "| Tier 3 | Remaining suppliers (5% of S3) | Sector initiatives | Industry collaboration, awareness |",
            ])

        return "\n".join(lines)

    def _md_offset_removal(self, data: Dict[str, Any]) -> str:
        offset = data.get("offset_strategy", {})

        return (
            f"## 7. Offset & Removal Strategy\n\n"
            f"### Principles\n\n"
            f"- Offsets are **not** used for achieving interim reduction targets\n"
            f"- Carbon removals used only for neutralizing residual emissions at net-zero\n"
            f"- All credits must meet ICVCM Core Carbon Principles\n"
            f"- Minimum **{_pct(offset.get('removal_share_pct', 50))}** carbon removals by net-zero date\n\n"
            f"### Strategy Details\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Estimated Residual Emissions | {_dec_comma(offset.get('residual_tco2e', 0))} tCO2e |\n"
            f"| Maximum Offset Volume | {_dec_comma(offset.get('max_offset_tco2e', 0))} tCO2e/year |\n"
            f"| Removal Target | {_pct(offset.get('removal_share_pct', 50))} of neutralization |\n"
            f"| Registry | {offset.get('registry', 'Verra VCS / Gold Standard')} |\n"
            f"| Quality Standard | {offset.get('quality_standard', 'ICVCM Core Carbon Principles')} |\n"
            f"| Permanence Requirement | {offset.get('permanence', '100+ years')} |\n"
            f"| Additionality | {offset.get('additionality', 'Verified per methodology')} |"
        )

    def _md_climate_finance(self, data: Dict[str, Any]) -> str:
        budget = data.get("budget", {})
        items = budget.get("items", [])
        total = budget.get("total_usd", 0)

        lines = [
            "## 8. Climate Finance Plan\n",
            f"**Total Planned Investment:** ${_dec_comma(total)}  \n"
            f"**Funding Sources:** {budget.get('funding_sources', 'Internal CAPEX, green bonds, climate finance')}\n",
        ]

        if items:
            lines.extend([
                "### Budget Allocation\n",
                "| Category | Year 1 | Year 2 | Year 3 | Year 4-5 | Total |",
                "|----------|-------:|-------:|-------:|--------:|------:|",
            ])
            for item in items:
                lines.append(
                    f"| {item.get('category', '-')} "
                    f"| ${_dec_comma(item.get('year1_usd', 0))} "
                    f"| ${_dec_comma(item.get('year2_usd', 0))} "
                    f"| ${_dec_comma(item.get('year3_usd', 0))} "
                    f"| ${_dec_comma(item.get('year4_5_usd', 0))} "
                    f"| ${_dec_comma(item.get('total_usd', 0))} |"
                )
        else:
            lines.append("_Budget allocation to be finalized during implementation planning._")

        roi = budget.get("roi_analysis", {})
        if roi:
            lines.extend([
                "\n### ROI Analysis\n",
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Payback Period | {roi.get('payback_years', 'N/A')} years |\n"
                f"| Internal Rate of Return | {_pct(roi.get('irr_pct', 0))} |\n"
                f"| Net Present Value | ${_dec_comma(roi.get('npv_usd', 0))} |\n"
                f"| Carbon Price Assumption | ${_dec(roi.get('carbon_price_usd', 0))} /tCO2e |",
            ])

        return "\n".join(lines)

    def _md_reporting_verification(self, data: Dict[str, Any]) -> str:
        reporting = data.get("reporting", {})
        schedule = reporting.get("schedule", [])

        lines = [
            "## 9. Reporting & Verification\n",
            f"**Reporting Standard:** {reporting.get('standard', 'Race to Zero + GHG Protocol')}  \n"
            f"**Disclosure Channel:** {reporting.get('channel', 'CDP + Annual Report')}  \n"
            f"**Verification Level:** {reporting.get('assurance_level', 'Limited assurance')}\n",
        ]

        if schedule:
            lines.extend([
                "### Reporting Schedule\n",
                "| Year | Deliverable | Audience | Verification |",
                "|:----:|-------------|----------|:------------:|",
            ])
            for s in schedule:
                lines.append(
                    f"| {s.get('year', '-')} | {s.get('deliverable', '-')} "
                    f"| {s.get('audience', '-')} | {s.get('verification', '-')} |"
                )
        else:
            lines.extend([
                "### Annual Reporting Cycle\n",
                "| Quarter | Activity | Output |",
                "|:-------:|----------|--------|",
                "| Q1 | Data collection and GHG inventory | Draft inventory report |",
                "| Q2 | Third-party verification | Verification statement |",
                "| Q3 | Race to Zero progress report | Published progress report |",
                "| Q4 | Target review and plan update | Updated action plan |",
            ])

        return "\n".join(lines)

    def _md_just_transition(self, data: Dict[str, Any]) -> str:
        jt = data.get("just_transition", {})
        pillars = jt.get("pillars", [])

        lines = [
            "## 10. Just Transition Plan\n",
            f"**Commitment:** {jt.get('commitment', 'Ensure fair and equitable transition for all stakeholders')}\n",
        ]

        if pillars:
            lines.extend([
                "### Transition Pillars\n",
                "| # | Pillar | Objective | Actions | Timeline |",
                "|---|--------|-----------|---------|:--------:|",
            ])
            for i, p in enumerate(pillars, 1):
                lines.append(
                    f"| {i} | {p.get('pillar', '-')} | {p.get('objective', '-')} "
                    f"| {p.get('actions', '-')} | {p.get('timeline', '-')} |"
                )
        else:
            lines.extend([
                "### Key Considerations\n",
                "- **Workforce**: Reskilling and retraining programs for affected workers",
                "- **Communities**: Support for communities dependent on transitioning industries",
                "- **Supply Chain**: Fair terms for suppliers adapting to new requirements",
                "- **Equity**: Ensure costs and benefits are distributed equitably",
                "- **Stakeholder Engagement**: Meaningful dialogue with affected groups",
            ])

        return "\n".join(lines)

    def _md_gantt_chart(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = ["## Appendix A: Gantt Chart / Timeline\n"]

        if milestones:
            lines.extend([
                "| Task | Owner | Start | End | Dependencies | Status |",
                "|------|-------|:-----:|:---:|:------------:|:------:|",
            ])
            for ms in milestones:
                lines.append(
                    f"| {ms.get('task', ms.get('milestone', '-'))} "
                    f"| {ms.get('owner', '-')} "
                    f"| {ms.get('start', '-')} "
                    f"| {ms.get('end', '-')} "
                    f"| {ms.get('dependencies', '-')} "
                    f"| {ms.get('status', 'Planned')} |"
                )

            # ASCII timeline
            lines.append("\n### Visual Timeline\n")
            lines.append("```")
            lines.append("2025  2026  2027  2028  2029  2030  2031  2032  ...  2050")
            lines.append("|-----|-----|-----|-----|-----|-----|-----|-----|     |")
            for ms in milestones[:10]:
                task = ms.get("task", ms.get("milestone", ""))[:30]
                lines.append(f"|{'=' * 5}> {task}")
            lines.append("```")
        else:
            lines.append("_Detailed timeline to be developed during implementation phase._")

        return "\n".join(lines)

    def _md_budget_allocation(self, data: Dict[str, Any]) -> str:
        budget = data.get("budget", {})
        items = budget.get("items", [])

        lines = ["## Appendix B: Budget Allocation\n"]

        if items:
            total = sum(it.get("total_usd", 0) for it in items)
            lines.extend([
                f"**Total Investment:** ${_dec_comma(total)}\n",
                "| # | Category | Description | Amount (USD) | % of Total |",
                "|---|----------|-------------|-------------:|:----------:|",
            ])
            for i, item in enumerate(items, 1):
                item_total = item.get("total_usd", 0)
                lines.append(
                    f"| {i} | {item.get('category', '-')} "
                    f"| {item.get('description', '-')} "
                    f"| ${_dec_comma(item_total)} "
                    f"| {_pct(_safe_div(item_total, max(total, 1)) * 100)} |"
                )
        else:
            lines.append("_Budget details to be finalized._")

        return "\n".join(lines)

    def _md_risk_mitigation(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", self.DEFAULT_RISKS)
        lines = [
            "## Appendix C: Risk Mitigation Strategies\n",
            "| # | Risk | Category | Likelihood | Impact | Mitigation |",
            "|---|------|----------|:----------:|:------:|------------|",
        ]

        for i, risk in enumerate(risks, 1):
            lines.append(
                f"| {i} | {risk.get('risk', '-')} "
                f"| {risk.get('category', '-')} "
                f"| {risk.get('likelihood', '-')} "
                f"| {risk.get('impact', '-')} "
                f"| {risk.get('mitigation', '-')} |"
            )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Action plan structure per Race to Zero Interpretation Guide.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".gantt-bar{background:#43a047;height:20px;border-radius:4px;margin:2px 0;"
            "display:inline-block;min-width:40px;}"
            ".risk-high{color:#d32f2f;font-weight:600;}"
            ".risk-medium{color:#f57c00;font-weight:600;}"
            ".risk-low{color:#388e3c;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Action Plan</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Sector:</strong> {data.get("sector", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        total_capex = sum(l.get("capex_usd", 0) for l in levers)
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Baseline</div>'
            f'<div class="card-value">{_dec_comma(baseline.get("total_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Interim Target</div>'
            f'<div class="card-value">{_pct(interim.get("reduction_pct", 50))}</div>by {interim.get("year", 2030)}</div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero</div>'
            f'<div class="card-value">{longterm.get("year", 2050)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Investment</div>'
            f'<div class="card-value">${_dec_comma(total_capex)}</div></div>\n'
            f'</div>'
        )

    def _html_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        roles = gov.get("roles", [])
        rows = ""
        for role in roles:
            rows += (f'<tr><td>{role.get("role", "-")}</td><td>{role.get("name", "-")}</td>'
                     f'<td>{role.get("responsibility", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Governance roles to be defined</em></td></tr>'
        return (
            f'<h2>2. Governance & Oversight</h2>\n'
            f'<table><tr><th>Role</th><th>Name</th><th>Responsibility</th></tr>\n{rows}</table>'
        )

    def _html_baseline(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        total = b.get("total_tco2e", 0)
        return (
            f'<h2>3. Baseline Emissions</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(b.get("scope1_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(b.get("scope2_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(b.get("scope3_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Total</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>tCO2e</div>\n'
            f'</div>'
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        return (
            f'<h2>4. Science-Based Targets</h2>\n'
            f'<table><tr><th>Parameter</th><th>Interim</th><th>Long-Term</th></tr>\n'
            f'<tr><td>Year</td><td>{interim.get("year", 2030)}</td><td>{longterm.get("year", 2050)}</td></tr>\n'
            f'<tr><td>Reduction</td><td>{_pct(interim.get("reduction_pct", 50))}</td>'
            f'<td>{_pct(longterm.get("min_reduction_pct", 90))}</td></tr>\n'
            f'<tr><td>Scope</td><td>{interim.get("scope_coverage", "S1+S2")}</td>'
            f'<td>{longterm.get("scope_coverage", "S1+S2+S3")}</td></tr>\n'
            f'</table>'
        )

    def _html_roadmap(self, data: Dict[str, Any]) -> str:
        levers = data.get("decarbonization_levers", self.DEFAULT_LEVERS)
        rows = ""
        for lever in levers:
            rows += (
                f'<tr><td>{lever.get("lever", "-")}</td><td>{lever.get("scope", "-")}</td>'
                f'<td>{_pct(lever.get("potential_pct", 0))}</td>'
                f'<td>${_dec_comma(lever.get("capex_usd", 0))}</td></tr>\n'
            )
        return (
            f'<h2>5. Decarbonization Roadmap</h2>\n'
            f'<table><tr><th>Lever</th><th>Scope</th><th>Reduction</th><th>CAPEX</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope3(self, data: Dict[str, Any]) -> str:
        s3 = data.get("scope3_engagement", {})
        strategies = s3.get("strategies", [])
        rows = ""
        for strat in strategies:
            rows += (
                f'<tr><td>{strat.get("category", "-")}</td><td>{strat.get("strategy", "-")}</td>'
                f'<td>{_pct(strat.get("coverage_pct", 0))}</td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="3"><em>Engagement strategies to be defined</em></td></tr>'
        return (
            f'<h2>6. Scope 3 Engagement</h2>\n'
            f'<table><tr><th>Category</th><th>Strategy</th><th>Coverage</th></tr>\n{rows}</table>'
        )

    def _html_offset(self, data: Dict[str, Any]) -> str:
        offset = data.get("offset_strategy", {})
        return (
            f'<h2>7. Offset & Removal Strategy</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Residual Emissions</div>'
            f'<div class="card-value">{_dec_comma(offset.get("residual_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Removal Target</div>'
            f'<div class="card-value">{_pct(offset.get("removal_share_pct", 50))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Quality</div>'
            f'<div class="card-value">ICVCM CCP</div></div>\n'
            f'</div>'
        )

    def _html_finance(self, data: Dict[str, Any]) -> str:
        budget = data.get("budget", {})
        total = budget.get("total_usd", 0)
        items = budget.get("items", [])
        rows = ""
        for item in items:
            rows += (
                f'<tr><td>{item.get("category", "-")}</td>'
                f'<td>${_dec_comma(item.get("total_usd", 0))}</td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="2"><em>Budget to be finalized</em></td></tr>'
        return (
            f'<h2>8. Climate Finance</h2>\n'
            f'<p><strong>Total Investment:</strong> ${_dec_comma(total)}</p>\n'
            f'<table><tr><th>Category</th><th>Total (USD)</th></tr>\n{rows}</table>'
        )

    def _html_reporting(self, data: Dict[str, Any]) -> str:
        reporting = data.get("reporting", {})
        return (
            f'<h2>9. Reporting & Verification</h2>\n'
            f'<table><tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Standard</td><td>{reporting.get("standard", "Race to Zero + GHG Protocol")}</td></tr>\n'
            f'<tr><td>Disclosure</td><td>{reporting.get("channel", "CDP + Annual Report")}</td></tr>\n'
            f'<tr><td>Assurance</td><td>{reporting.get("assurance_level", "Limited assurance")}</td></tr>\n'
            f'</table>'
        )

    def _html_just_transition(self, data: Dict[str, Any]) -> str:
        jt = data.get("just_transition", {})
        pillars = jt.get("pillars", [])
        rows = ""
        for p in pillars:
            rows += f'<tr><td>{p.get("pillar", "-")}</td><td>{p.get("objective", "-")}</td></tr>\n'
        if not rows:
            rows = '<tr><td colspan="2"><em>Just transition plan in development</em></td></tr>'
        return (
            f'<h2>10. Just Transition</h2>\n'
            f'<table><tr><th>Pillar</th><th>Objective</th></tr>\n{rows}</table>'
        )

    def _html_gantt(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for ms in milestones:
            rows += (
                f'<tr><td>{ms.get("task", ms.get("milestone", "-"))}</td>'
                f'<td>{ms.get("start", "-")}</td><td>{ms.get("end", "-")}</td>'
                f'<td>{ms.get("status", "Planned")}</td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="4"><em>Timeline to be developed</em></td></tr>'
        return (
            f'<h2>Appendix A: Timeline</h2>\n'
            f'<table><tr><th>Task</th><th>Start</th><th>End</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_budget(self, data: Dict[str, Any]) -> str:
        items = data.get("budget", {}).get("items", [])
        rows = ""
        for item in items:
            rows += (
                f'<tr><td>{item.get("category", "-")}</td>'
                f'<td>${_dec_comma(item.get("year1_usd", 0))}</td>'
                f'<td>${_dec_comma(item.get("year2_usd", 0))}</td>'
                f'<td>${_dec_comma(item.get("total_usd", 0))}</td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="4"><em>Budget details pending</em></td></tr>'
        return (
            f'<h2>Appendix B: Budget</h2>\n'
            f'<table><tr><th>Category</th><th>Year 1</th><th>Year 2</th><th>Total</th></tr>\n{rows}</table>'
        )

    def _html_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", self.DEFAULT_RISKS)
        rows = ""
        for risk in risks:
            impact = risk.get("impact", "Medium")
            css_class = "risk-high" if impact == "High" else ("risk-medium" if impact == "Medium" else "risk-low")
            rows += (
                f'<tr><td>{risk.get("risk", "-")}</td>'
                f'<td>{risk.get("likelihood", "-")}</td>'
                f'<td class="{css_class}">{impact}</td>'
                f'<td>{risk.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            f'<h2>Appendix C: Risk Register</h2>\n'
            f'<table><tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
