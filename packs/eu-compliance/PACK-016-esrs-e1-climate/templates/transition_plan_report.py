# -*- coding: utf-8 -*-
"""
TransitionPlanReportTemplate - ESRS E1-1 Transition Plan Report

Renders climate transition plan overview, Paris Agreement alignment,
decarbonization levers, action timelines, CapEx allocation, locked-in
emissions analysis, and gap assessment per ESRS E1-1.

Sections:
    1. Plan Overview
    2. Scenario Alignment
    3. Levers
    4. Action Timeline
    5. CapEx Allocation
    6. Locked-In Emissions
    7. Gap Analysis

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "plan_overview",
    "scenario_alignment",
    "levers",
    "action_timeline",
    "capex_allocation",
    "locked_in_emissions",
    "gap_analysis",
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TransitionPlanReportTemplate:
    """
    Climate transition plan report template per ESRS E1-1.

    Renders Paris-aligned transition plan with scenario analysis, lever
    mapping, phased action timeline, CapEx/OpEx allocation, locked-in
    emissions assessment, and residual gap analysis.

    Example:
        >>> tpl = TransitionPlanReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TransitionPlanReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {}
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        result["generated_at"] = self.generated_at.isoformat()
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("transition_plan"):
            errors.append("transition_plan data is required for E1-1")
        if "scenario_alignment" not in data:
            warnings.append("scenario_alignment missing; will default to empty")
        if "decarbonization_levers" not in data:
            warnings.append("decarbonization_levers missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render transition plan report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_plan_overview(data),
            self._md_scenario_alignment(data),
            self._md_levers(data),
            self._md_action_timeline(data),
            self._md_capex(data),
            self._md_locked_in(data),
            self._md_gap_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render transition plan report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_plan_overview(data),
            self._html_scenario_alignment(data),
            self._html_levers(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Transition Plan Report - ESRS E1-1</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render transition plan report as JSON."""
        self.generated_at = _utcnow()
        plan = data.get("transition_plan", {})
        result = {
            "template": "transition_plan_report",
            "esrs_reference": "E1-1",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "has_transition_plan": plan.get("adopted", False),
            "adoption_date": plan.get("adoption_date", ""),
            "target_year": plan.get("target_year", ""),
            "scenario_reference": data.get("scenario_reference", ""),
            "levers_count": len(data.get("decarbonization_levers", [])),
            "total_capex_eur": data.get("total_capex_eur", 0.0),
            "locked_in_tco2e": data.get("locked_in_tco2e", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_plan_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build plan overview section."""
        plan = data.get("transition_plan", {})
        return {
            "title": "Transition Plan Overview",
            "adopted": plan.get("adopted", False),
            "adoption_date": plan.get("adoption_date", ""),
            "approved_by": plan.get("approved_by", ""),
            "target_year": plan.get("target_year", ""),
            "net_zero_year": plan.get("net_zero_year", ""),
            "scope_coverage": plan.get("scope_coverage", []),
            "key_assumptions": plan.get("key_assumptions", []),
            "update_frequency": plan.get("update_frequency", "annual"),
        }

    def _section_scenario_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scenario alignment section."""
        scenario = data.get("scenario_alignment", {})
        return {
            "title": "Climate Scenario Alignment",
            "reference_scenario": scenario.get("reference_scenario", ""),
            "temperature_target": scenario.get("temperature_target", ""),
            "scenario_provider": scenario.get("scenario_provider", ""),
            "alignment_score": round(scenario.get("alignment_score", 0.0), 2),
            "pathway_gap_tco2e": round(scenario.get("pathway_gap_tco2e", 0.0), 2),
            "paris_aligned": scenario.get("paris_aligned", False),
            "sectoral_pathway": scenario.get("sectoral_pathway", ""),
        }

    def _section_levers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build decarbonization levers section."""
        levers = data.get("decarbonization_levers", [])
        total_abatement = sum(lv.get("abatement_tco2e", 0.0) for lv in levers)
        return {
            "title": "Decarbonization Levers",
            "total_abatement_tco2e": round(total_abatement, 2),
            "lever_count": len(levers),
            "levers": [
                {
                    "name": lv.get("name", ""),
                    "category": lv.get("category", ""),
                    "abatement_tco2e": round(lv.get("abatement_tco2e", 0.0), 2),
                    "cost_eur": round(lv.get("cost_eur", 0.0), 2),
                    "implementation_year": lv.get("implementation_year", ""),
                    "readiness_level": lv.get("readiness_level", ""),
                }
                for lv in levers
            ],
        }

    def _section_action_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build action timeline section."""
        actions = data.get("actions_timeline", [])
        return {
            "title": "Action Timeline",
            "short_term": [a for a in actions if a.get("horizon") == "short_term"],
            "medium_term": [a for a in actions if a.get("horizon") == "medium_term"],
            "long_term": [a for a in actions if a.get("horizon") == "long_term"],
            "total_actions": len(actions),
        }

    def _section_capex_allocation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build CapEx allocation section."""
        capex = data.get("capex_allocation", {})
        items = capex.get("items", [])
        total = sum(it.get("amount_eur", 0.0) for it in items)
        return {
            "title": "CapEx Allocation for Climate Transition",
            "total_capex_eur": round(total, 2),
            "taxonomy_aligned_pct": round(capex.get("taxonomy_aligned_pct", 0.0), 1),
            "items": [
                {
                    "category": it.get("category", ""),
                    "amount_eur": round(it.get("amount_eur", 0.0), 2),
                    "percentage": (
                        round(it.get("amount_eur", 0.0) / total * 100, 1)
                        if total > 0 else 0.0
                    ),
                    "timeline": it.get("timeline", ""),
                }
                for it in items
            ],
            "opex_related_eur": round(capex.get("opex_related_eur", 0.0), 2),
        }

    def _section_locked_in_emissions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build locked-in emissions section."""
        locked = data.get("locked_in_emissions", {})
        return {
            "title": "Locked-In GHG Emissions",
            "total_locked_tco2e": round(locked.get("total_tco2e", 0.0), 2),
            "asset_categories": locked.get("asset_categories", []),
            "remaining_useful_life_years": locked.get("remaining_useful_life_years", 0),
            "stranded_asset_risk": locked.get("stranded_asset_risk", "low"),
            "phase_out_plan": locked.get("phase_out_plan", ""),
        }

    def _section_gap_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap analysis section."""
        gap = data.get("gap_analysis", {})
        return {
            "title": "Transition Gap Analysis",
            "target_reduction_tco2e": round(gap.get("target_reduction_tco2e", 0.0), 2),
            "planned_reduction_tco2e": round(gap.get("planned_reduction_tco2e", 0.0), 2),
            "residual_gap_tco2e": round(gap.get("residual_gap_tco2e", 0.0), 2),
            "gap_percentage": round(gap.get("gap_percentage", 0.0), 1),
            "mitigation_options": gap.get("mitigation_options", []),
            "confidence_level": gap.get("confidence_level", "medium"),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# Transition Plan Report - ESRS E1-1\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-1 Transition Plan for Climate Change Mitigation"
        )

    def _md_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render plan overview markdown."""
        sec = self._section_plan_overview(data)
        status = "Adopted" if sec["adopted"] else "Not Yet Adopted"
        scopes = ", ".join(sec["scope_coverage"]) if sec["scope_coverage"] else "N/A"
        return (
            f"## {sec['title']}\n\n"
            f"**Status:** {status}  \n"
            f"**Adoption Date:** {sec['adoption_date']}  \n"
            f"**Approved By:** {sec['approved_by']}  \n"
            f"**Target Year:** {sec['target_year']}  \n"
            f"**Net-Zero Year:** {sec['net_zero_year']}  \n"
            f"**Scope Coverage:** {scopes}"
        )

    def _md_scenario_alignment(self, data: Dict[str, Any]) -> str:
        """Render scenario alignment markdown."""
        sec = self._section_scenario_alignment(data)
        aligned = "Yes" if sec["paris_aligned"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**Reference Scenario:** {sec['reference_scenario']}  \n"
            f"**Temperature Target:** {sec['temperature_target']}  \n"
            f"**Provider:** {sec['scenario_provider']}  \n"
            f"**Alignment Score:** {sec['alignment_score']:.2f}  \n"
            f"**Paris-Aligned:** {aligned}"
        )

    def _md_levers(self, data: Dict[str, Any]) -> str:
        """Render decarbonization levers markdown."""
        sec = self._section_levers(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Abatement:** {sec['total_abatement_tco2e']:,.2f} tCO2e\n",
            "| Lever | Category | Abatement (tCO2e) | Cost (EUR) |",
            "|-------|----------|------------------:|----------:|",
        ]
        for lv in sec["levers"]:
            lines.append(
                f"| {lv['name']} | {lv['category']} "
                f"| {lv['abatement_tco2e']:,.2f} | {lv['cost_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_action_timeline(self, data: Dict[str, Any]) -> str:
        """Render action timeline markdown."""
        sec = self._section_action_timeline(data)
        lines = [f"## {sec['title']}\n"]
        for horizon, label in [
            ("short_term", "Short-Term (0-2 years)"),
            ("medium_term", "Medium-Term (2-5 years)"),
            ("long_term", "Long-Term (5+ years)"),
        ]:
            items = sec[horizon]
            lines.append(f"### {label}")
            if items:
                for a in items:
                    lines.append(f"- {a.get('name', '')} ({a.get('status', '')})")
            else:
                lines.append("- No actions defined")
            lines.append("")
        return "\n".join(lines)

    def _md_capex(self, data: Dict[str, Any]) -> str:
        """Render CapEx allocation markdown."""
        sec = self._section_capex_allocation(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total CapEx:** EUR {sec['total_capex_eur']:,.2f}  \n"
            f"**Taxonomy-Aligned:** {sec['taxonomy_aligned_pct']:.1f}%\n",
            "| Category | EUR | % |",
            "|----------|----:|--:|",
        ]
        for it in sec["items"]:
            lines.append(
                f"| {it['category']} | {it['amount_eur']:,.2f} | {it['percentage']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_locked_in(self, data: Dict[str, Any]) -> str:
        """Render locked-in emissions markdown."""
        sec = self._section_locked_in_emissions(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Total Locked-In:** {sec['total_locked_tco2e']:,.2f} tCO2e  \n"
            f"**Remaining Life:** {sec['remaining_useful_life_years']} years  \n"
            f"**Stranded Asset Risk:** {sec['stranded_asset_risk']}"
        )

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis markdown."""
        sec = self._section_gap_analysis(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Target Reduction:** {sec['target_reduction_tco2e']:,.2f} tCO2e  \n"
            f"**Planned Reduction:** {sec['planned_reduction_tco2e']:,.2f} tCO2e  \n"
            f"**Residual Gap:** {sec['residual_gap_tco2e']:,.2f} tCO2e "
            f"({sec['gap_percentage']:.1f}%)  \n"
            f"**Confidence:** {sec['confidence_level']}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-016 ESRS E1 Climate Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:900px;margin:auto}"
            "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}"
            "h2{color:#2d7a4f;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#f0f7f3}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>Transition Plan Report - ESRS E1-1</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render plan overview HTML."""
        sec = self._section_plan_overview(data)
        status = "Adopted" if sec["adopted"] else "Not Yet Adopted"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Status: <strong>{status}</strong> | "
            f"Target: {sec['target_year']}</p>"
        )

    def _html_scenario_alignment(self, data: Dict[str, Any]) -> str:
        """Render scenario alignment HTML."""
        sec = self._section_scenario_alignment(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Reference: {sec['reference_scenario']} | "
            f"Temperature: {sec['temperature_target']}</p>"
        )

    def _html_levers(self, data: Dict[str, Any]) -> str:
        """Render decarbonization levers HTML."""
        sec = self._section_levers(data)
        rows = "".join(
            f"<tr><td>{lv['name']}</td><td>{lv['category']}</td>"
            f"<td>{lv['abatement_tco2e']:,.2f}</td></tr>"
            for lv in sec["levers"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Lever</th><th>Category</th><th>tCO2e</th></tr>"
            f"{rows}</table>"
        )
