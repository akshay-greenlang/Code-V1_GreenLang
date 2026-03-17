# -*- coding: utf-8 -*-
"""
ClimateTransitionReportTemplate - CSDDD Article 22 Climate Transition Plan Report

Renders the climate transition plan assessment covering Paris Agreement alignment,
GHG emission reduction targets by scope, decarbonisation pathway analysis, and
implementation milestone tracking per CSDDD Article 22.

Regulatory References:
    - Directive (EU) 2024/1760, Article 22 (Combating Climate Change)
    - Paris Agreement (2015), Article 2.1(a) - 1.5C temperature goal
    - Regulation (EU) 2021/1119 (European Climate Law)
    - ESRS E1-1 (Transition Plan for Climate Change Mitigation)
    - IPCC AR6 emissions pathways

Sections:
    1. Plan Summary - Overall transition plan status and ambition
    2. Target Analysis - GHG reduction targets assessment
    3. Scope Breakdown - Emissions by Scope 1/2/3 with reduction paths
    4. Pathway Assessment - Decarbonisation trajectory vs benchmarks
    5. Paris Alignment - 1.5C/2C alignment scoring
    6. Implementation Status - Progress against plan elements
    7. Milestones - Key milestones and deadlines

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "plan_summary",
    "target_analysis",
    "scope_breakdown",
    "pathway_assessment",
    "paris_alignment",
    "implementation_status",
    "milestones",
]

# Paris-aligned emission reduction benchmarks (% from base year)
_PARIS_15C_BENCHMARKS: Dict[int, float] = {
    2025: 25.0,
    2030: 45.0,
    2035: 60.0,
    2040: 75.0,
    2045: 87.0,
    2050: 95.0,
}

_PARIS_2C_BENCHMARKS: Dict[int, float] = {
    2025: 15.0,
    2030: 25.0,
    2035: 40.0,
    2040: 55.0,
    2045: 70.0,
    2050: 85.0,
}


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


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


def _interpolate_benchmark(
    benchmarks: Dict[int, float], year: int
) -> float:
    """Linearly interpolate benchmark value for a given year."""
    years = sorted(benchmarks.keys())
    if year <= years[0]:
        return benchmarks[years[0]]
    if year >= years[-1]:
        return benchmarks[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            y1, y2 = years[i], years[i + 1]
            v1, v2 = benchmarks[y1], benchmarks[y2]
            fraction = (year - y1) / (y2 - y1)
            return round(v1 + fraction * (v2 - v1), 1)
    return 0.0


class ClimateTransitionReportTemplate:
    """
    CSDDD Article 22 Climate Transition Plan Report.

    Renders a comprehensive climate transition plan assessment including
    GHG reduction target analysis, scope-level pathway tracking, Paris
    Agreement alignment scoring against 1.5C and 2C benchmarks, and
    implementation milestone tracking. All calculations are deterministic
    using IPCC AR6 benchmarks.

    Example:
        >>> tpl = ClimateTransitionReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimateTransitionReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        report_id = _new_uuid()
        result: Dict[str, Any] = {"report_id": report_id}
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
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if "transition_plan" not in data:
            errors.append("transition_plan is required for climate transition report")
        if "base_year" not in data:
            warnings.append("base_year missing; pathway analysis will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render climate transition report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_plan_summary(data),
            self._md_target_analysis(data),
            self._md_scope_breakdown(data),
            self._md_pathway_assessment(data),
            self._md_paris_alignment(data),
            self._md_implementation_status(data),
            self._md_milestones(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render climate transition report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_plan_summary(data),
            self._html_targets(data),
            self._html_scopes(data),
            self._html_paris_alignment(data),
            self._html_milestones(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Transition Plan - CSDDD Art 22</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render climate transition report as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "climate_transition_report",
            "directive_reference": "Directive (EU) 2024/1760, Art 22",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "plan_summary": self._section_plan_summary(data),
            "target_analysis": self._section_target_analysis(data),
            "scope_breakdown": self._section_scope_breakdown(data),
            "pathway_assessment": self._section_pathway_assessment(data),
            "paris_alignment": self._section_paris_alignment(data),
            "implementation_status": self._section_implementation_status(data),
            "milestones": self._section_milestones(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_plan_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build transition plan summary section."""
        plan = data.get("transition_plan", {})
        base = data.get("base_year", {})
        alignment = self._calculate_paris_alignment(data)
        return {
            "title": "Transition Plan Summary",
            "plan_exists": plan.get("exists", False),
            "plan_adopted_date": plan.get("adopted_date", ""),
            "approved_by_board": plan.get("board_approved", False),
            "net_zero_target_year": plan.get("net_zero_year", ""),
            "interim_target_year": plan.get("interim_target_year", ""),
            "interim_reduction_pct": round(plan.get("interim_reduction_pct", 0.0), 1),
            "base_year": base.get("year", ""),
            "base_year_emissions_tco2e": round(base.get("total_tco2e", 0.0), 2),
            "current_emissions_tco2e": round(
                data.get("current_emissions_tco2e", 0.0), 2
            ),
            "total_reduction_achieved_pct": self._calc_total_reduction(data),
            "paris_alignment_status": alignment["alignment_status"],
            "paris_alignment_score_pct": alignment["alignment_score"],
            "sbti_validated": plan.get("sbti_validated", False),
            "capex_allocated_eur": round(plan.get("capex_allocated_eur", 0.0), 2),
        }

    def _section_target_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build target analysis section."""
        targets = data.get("climate_targets", [])
        entries: List[Dict[str, Any]] = []
        for t in targets:
            base_val = t.get("base_year_tco2e", 0.0)
            current_val = t.get("current_tco2e", 0.0)
            actual_reduction = (
                round((base_val - current_val) / base_val * 100, 1)
                if base_val > 0 else 0.0
            )
            target_reduction = t.get("target_reduction_pct", 0.0)
            entries.append({
                "target_name": t.get("name", ""),
                "scope_coverage": t.get("scope_coverage", []),
                "target_type": t.get("target_type", "absolute"),
                "base_year": t.get("base_year", ""),
                "target_year": t.get("target_year", ""),
                "target_reduction_pct": round(target_reduction, 1),
                "actual_reduction_pct": actual_reduction,
                "gap_pct": round(target_reduction - actual_reduction, 1),
                "on_track": actual_reduction >= (target_reduction * 0.8),
                "sbti_approved": t.get("sbti_approved", False),
            })
        on_track_count = sum(1 for e in entries if e["on_track"])
        return {
            "title": "GHG Reduction Target Analysis",
            "total_targets": len(entries),
            "targets_on_track": on_track_count,
            "targets": entries,
        }

    def _section_scope_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scope-level emissions breakdown section."""
        base = data.get("base_year", {})
        current = data.get("current_emissions", {})
        scopes: List[Dict[str, Any]] = []
        for scope_key, scope_label in [
            ("scope1", "Scope 1 - Direct"),
            ("scope2", "Scope 2 - Indirect (Energy)"),
            ("scope3", "Scope 3 - Value Chain"),
        ]:
            base_val = base.get(f"{scope_key}_tco2e", 0.0)
            current_val = current.get(f"{scope_key}_tco2e", 0.0)
            reduction = (
                round((base_val - current_val) / base_val * 100, 1)
                if base_val > 0 else 0.0
            )
            scopes.append({
                "scope": scope_label,
                "scope_key": scope_key,
                "base_year_tco2e": round(base_val, 2),
                "current_tco2e": round(current_val, 2),
                "reduction_pct": reduction,
                "target_reduction_pct": round(
                    current.get(f"{scope_key}_target_pct", 0.0), 1
                ),
                "key_reduction_levers": current.get(
                    f"{scope_key}_levers", []
                ),
            })
        return {
            "title": "Scope-Level Emissions Breakdown",
            "scopes": scopes,
            "total_base_tco2e": round(base.get("total_tco2e", 0.0), 2),
            "total_current_tco2e": round(
                sum(s["current_tco2e"] for s in scopes), 2
            ),
        }

    def _section_pathway_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build decarbonisation pathway assessment section."""
        pathway_points = data.get("pathway_points", [])
        entries: List[Dict[str, Any]] = []
        for point in pathway_points:
            year = point.get("year", 0)
            actual_reduction = point.get("reduction_pct", 0.0)
            benchmark_15c = _interpolate_benchmark(_PARIS_15C_BENCHMARKS, year)
            benchmark_2c = _interpolate_benchmark(_PARIS_2C_BENCHMARKS, year)
            entries.append({
                "year": year,
                "actual_reduction_pct": round(actual_reduction, 1),
                "benchmark_15c_pct": benchmark_15c,
                "benchmark_2c_pct": benchmark_2c,
                "gap_to_15c_pct": round(benchmark_15c - actual_reduction, 1),
                "gap_to_2c_pct": round(benchmark_2c - actual_reduction, 1),
                "aligned_15c": actual_reduction >= benchmark_15c,
                "aligned_2c": actual_reduction >= benchmark_2c,
            })
        return {
            "title": "Decarbonisation Pathway Assessment",
            "total_data_points": len(entries),
            "pathway_points": entries,
            "years_aligned_15c": sum(1 for e in entries if e["aligned_15c"]),
            "years_aligned_2c": sum(1 for e in entries if e["aligned_2c"]),
        }

    def _section_paris_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Paris Agreement alignment section."""
        alignment = self._calculate_paris_alignment(data)
        return {
            "title": "Paris Agreement Alignment Assessment",
            "alignment_status": alignment["alignment_status"],
            "alignment_score_pct": alignment["alignment_score"],
            "temperature_pathway": alignment["temperature_pathway"],
            "aligned_with_15c": alignment["aligned_15c"],
            "aligned_with_2c": alignment["aligned_2c"],
            "current_reduction_pct": alignment["current_reduction"],
            "required_15c_reduction_pct": alignment["required_15c"],
            "required_2c_reduction_pct": alignment["required_2c"],
            "gap_to_15c_pct": alignment["gap_15c"],
            "gap_to_2c_pct": alignment["gap_2c"],
            "eu_climate_law_compatible": alignment["eu_climate_law"],
        }

    def _section_implementation_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build implementation status section."""
        plan = data.get("transition_plan", {})
        elements = plan.get("implementation_elements", [])
        entries: List[Dict[str, Any]] = []
        for elem in elements:
            entries.append({
                "element": elem.get("element", ""),
                "description": elem.get("description", ""),
                "status": elem.get("status", "not_started"),
                "progress_pct": round(elem.get("progress_pct", 0.0), 1),
                "responsible": elem.get("responsible", ""),
                "budget_eur": round(elem.get("budget_eur", 0.0), 2),
                "spent_eur": round(elem.get("spent_eur", 0.0), 2),
                "start_date": elem.get("start_date", ""),
                "target_date": elem.get("target_date", ""),
            })
        avg_progress = (
            round(sum(e["progress_pct"] for e in entries) / len(entries), 1)
            if entries else 0.0
        )
        return {
            "title": "Implementation Status",
            "total_elements": len(entries),
            "average_progress_pct": avg_progress,
            "elements_completed": sum(
                1 for e in entries if e["status"] == "completed"
            ),
            "elements_in_progress": sum(
                1 for e in entries if e["status"] == "in_progress"
            ),
            "elements_not_started": sum(
                1 for e in entries if e["status"] == "not_started"
            ),
            "elements": entries,
            "total_budget_eur": round(sum(e["budget_eur"] for e in entries), 2),
            "total_spent_eur": round(sum(e["spent_eur"] for e in entries), 2),
        }

    def _section_milestones(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build milestones section."""
        milestones = data.get("climate_milestones", [])
        entries: List[Dict[str, Any]] = []
        for m in milestones:
            entries.append({
                "milestone_id": m.get("milestone_id", ""),
                "description": m.get("description", ""),
                "target_date": m.get("target_date", ""),
                "status": m.get("status", "pending"),
                "scope_coverage": m.get("scope_coverage", []),
                "emission_reduction_tco2e": round(
                    m.get("emission_reduction_tco2e", 0.0), 2
                ),
                "investment_eur": round(m.get("investment_eur", 0.0), 2),
                "key_actions": m.get("key_actions", []),
            })
        completed = sum(1 for e in entries if e["status"] == "completed")
        return {
            "title": "Key Milestones",
            "total_milestones": len(entries),
            "completed": completed,
            "on_track": sum(1 for e in entries if e["status"] == "on_track"),
            "at_risk": sum(1 for e in entries if e["status"] == "at_risk"),
            "delayed": sum(1 for e in entries if e["status"] == "delayed"),
            "completion_rate_pct": round(
                completed / len(entries) * 100, 1
            ) if entries else 0.0,
            "milestones": entries,
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _calc_total_reduction(self, data: Dict[str, Any]) -> float:
        """Calculate total GHG reduction percentage from base year."""
        base = data.get("base_year", {})
        base_total = base.get("total_tco2e", 0.0)
        current_total = data.get("current_emissions_tco2e", 0.0)
        if base_total <= 0:
            return 0.0
        return round((base_total - current_total) / base_total * 100, 1)

    def _calculate_paris_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Paris Agreement alignment scoring."""
        current_reduction = self._calc_total_reduction(data)
        reporting_year = data.get("reporting_year", 2025)
        if isinstance(reporting_year, str):
            try:
                reporting_year = int(reporting_year)
            except (ValueError, TypeError):
                reporting_year = 2025
        required_15c = _interpolate_benchmark(_PARIS_15C_BENCHMARKS, reporting_year)
        required_2c = _interpolate_benchmark(_PARIS_2C_BENCHMARKS, reporting_year)
        aligned_15c = current_reduction >= required_15c
        aligned_2c = current_reduction >= required_2c
        gap_15c = round(required_15c - current_reduction, 1) if not aligned_15c else 0.0
        gap_2c = round(required_2c - current_reduction, 1) if not aligned_2c else 0.0
        if aligned_15c:
            alignment_score = min(
                100.0, round(current_reduction / required_15c * 100, 1)
            )
            temp_pathway = "Well below 2C (1.5C aligned)"
            status = "Paris-Aligned (1.5C)"
        elif aligned_2c:
            alignment_score = round(current_reduction / required_15c * 100, 1)
            temp_pathway = "Below 2C"
            status = "Paris-Aligned (2C)"
        elif current_reduction > 0:
            alignment_score = round(current_reduction / required_2c * 100, 1)
            temp_pathway = "Above 2C"
            status = "Not Paris-Aligned"
        else:
            alignment_score = 0.0
            temp_pathway = "No reduction pathway"
            status = "Not Paris-Aligned"
        eu_55_target = _interpolate_benchmark(
            {2030: 55.0, 2050: 100.0}, reporting_year
        )
        eu_law = current_reduction >= eu_55_target * 0.5
        return {
            "alignment_status": status,
            "alignment_score": alignment_score,
            "temperature_pathway": temp_pathway,
            "aligned_15c": aligned_15c,
            "aligned_2c": aligned_2c,
            "current_reduction": current_reduction,
            "required_15c": required_15c,
            "required_2c": required_2c,
            "gap_15c": gap_15c,
            "gap_2c": gap_2c,
            "eu_climate_law": eu_law,
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Climate Transition Plan Report - CSDDD Art 22\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 22 | Paris Agreement"
        )

    def _md_plan_summary(self, data: Dict[str, Any]) -> str:
        """Render plan summary as markdown."""
        sec = self._section_plan_summary(data)
        plan_exists = "Yes" if sec["plan_exists"] else "No"
        board = "Yes" if sec["approved_by_board"] else "No"
        sbti = "Yes" if sec["sbti_validated"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Plan Exists | {plan_exists} |\n"
            f"| Board Approved | {board} |\n"
            f"| Net Zero Target Year | {sec['net_zero_target_year']} |\n"
            f"| Base Year | {sec['base_year']} |\n"
            f"| Base Year Emissions | {sec['base_year_emissions_tco2e']:,.2f} tCO2e |\n"
            f"| Current Emissions | {sec['current_emissions_tco2e']:,.2f} tCO2e |\n"
            f"| Reduction Achieved | {sec['total_reduction_achieved_pct']:.1f}% |\n"
            f"| Paris Alignment | {sec['paris_alignment_status']} |\n"
            f"| SBTi Validated | {sbti} |\n"
            f"| CapEx Allocated | EUR {sec['capex_allocated_eur']:,.2f} |"
        )

    def _md_target_analysis(self, data: Dict[str, Any]) -> str:
        """Render target analysis as markdown."""
        sec = self._section_target_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Targets:** {sec['total_targets']} | "
            f"**On Track:** {sec['targets_on_track']}\n",
            "| Target | Scope | Type | Target % | Actual % | Gap | On Track |",
            "|--------|-------|------|--------:|--------:|----:|:--------:|",
        ]
        for t in sec["targets"]:
            scopes = ", ".join(t["scope_coverage"])
            on_track = "Yes" if t["on_track"] else "No"
            lines.append(
                f"| {t['target_name'][:30]} | {scopes} | {t['target_type']} | "
                f"{t['target_reduction_pct']:.1f}% | {t['actual_reduction_pct']:.1f}% | "
                f"{t['gap_pct']:.1f}pp | {on_track} |"
            )
        return "\n".join(lines)

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        """Render scope breakdown as markdown."""
        sec = self._section_scope_breakdown(data)
        lines = [
            f"## {sec['title']}\n",
            "| Scope | Base Year (tCO2e) | Current (tCO2e) | Reduction |",
            "|-------|------------------:|----------------:|----------:|",
        ]
        for s in sec["scopes"]:
            lines.append(
                f"| {s['scope']} | {s['base_year_tco2e']:,.2f} | "
                f"{s['current_tco2e']:,.2f} | {s['reduction_pct']:.1f}% |"
            )
        lines.append(
            f"| **Total** | **{sec['total_base_tco2e']:,.2f}** | "
            f"**{sec['total_current_tco2e']:,.2f}** | |"
        )
        return "\n".join(lines)

    def _md_pathway_assessment(self, data: Dict[str, Any]) -> str:
        """Render pathway assessment as markdown."""
        sec = self._section_pathway_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Data Points:** {sec['total_data_points']} | "
            f"**1.5C Aligned Years:** {sec['years_aligned_15c']} | "
            f"**2C Aligned Years:** {sec['years_aligned_2c']}\n",
            "| Year | Actual % | 1.5C Bench % | 2C Bench % | 1.5C Gap | 2C Gap |",
            "|-----:|--------:|-----------:|---------:|-------:|------:|",
        ]
        for p in sec["pathway_points"]:
            lines.append(
                f"| {p['year']} | {p['actual_reduction_pct']:.1f}% | "
                f"{p['benchmark_15c_pct']:.1f}% | {p['benchmark_2c_pct']:.1f}% | "
                f"{p['gap_to_15c_pct']:.1f}pp | {p['gap_to_2c_pct']:.1f}pp |"
            )
        return "\n".join(lines)

    def _md_paris_alignment(self, data: Dict[str, Any]) -> str:
        """Render Paris alignment as markdown."""
        sec = self._section_paris_alignment(data)
        aligned_15 = "Yes" if sec["aligned_with_15c"] else "No"
        aligned_2 = "Yes" if sec["aligned_with_2c"] else "No"
        eu = "Yes" if sec["eu_climate_law_compatible"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"### Status: {sec['alignment_status']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Alignment Score | {sec['alignment_score_pct']:.1f}% |\n"
            f"| Temperature Pathway | {sec['temperature_pathway']} |\n"
            f"| Aligned with 1.5C | {aligned_15} |\n"
            f"| Aligned with 2C | {aligned_2} |\n"
            f"| Current Reduction | {sec['current_reduction_pct']:.1f}% |\n"
            f"| Required for 1.5C | {sec['required_15c_reduction_pct']:.1f}% |\n"
            f"| Required for 2C | {sec['required_2c_reduction_pct']:.1f}% |\n"
            f"| EU Climate Law Compatible | {eu} |"
        )

    def _md_implementation_status(self, data: Dict[str, Any]) -> str:
        """Render implementation status as markdown."""
        sec = self._section_implementation_status(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Average Progress:** {sec['average_progress_pct']:.1f}%  \n"
            f"**Completed:** {sec['elements_completed']} | "
            f"**In Progress:** {sec['elements_in_progress']} | "
            f"**Not Started:** {sec['elements_not_started']}\n",
            "| Element | Status | Progress | Budget (EUR) | Spent (EUR) |",
            "|---------|--------|--------:|------------:|----------:|",
        ]
        for e in sec["elements"]:
            lines.append(
                f"| {e['element'][:35]} | {e['status']} | "
                f"{e['progress_pct']:.1f}% | {e['budget_eur']:,.2f} | "
                f"{e['spent_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        """Render milestones as markdown."""
        sec = self._section_milestones(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_milestones']} | "
            f"**Completed:** {sec['completed']} | "
            f"**On Track:** {sec['on_track']} | "
            f"**At Risk:** {sec['at_risk']} | "
            f"**Delayed:** {sec['delayed']}\n",
            "| Milestone | Target Date | Status | Reduction (tCO2e) | Investment (EUR) |",
            "|-----------|-------------|--------|------------------:|-----------------:|",
        ]
        for m in sec["milestones"]:
            lines.append(
                f"| {m['description'][:35]} | {m['target_date']} | "
                f"{m['status']} | {m['emission_reduction_tco2e']:,.2f} | "
                f"{m['investment_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-019 CSDDD Readiness Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:.3em}"
            "h2{color:#283593;margin-top:1.5em}"
            "h3{color:#3949ab}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".aligned{color:#2e7d32;font-weight:bold}"
            ".not-aligned{color:#c62828;font-weight:bold}"
            ".on-track{color:#2e7d32}"
            ".delayed{color:#c62828}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Climate Transition Plan - CSDDD Art 22</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_plan_summary(self, data: Dict[str, Any]) -> str:
        """Render plan summary HTML."""
        sec = self._section_plan_summary(data)
        css = "aligned" if "Aligned" in sec["paris_alignment_status"] else "not-aligned"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css}'>Paris Alignment: {sec['paris_alignment_status']}</p>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Net Zero Target</td><td>{sec['net_zero_target_year']}</td></tr>"
            f"<tr><td>Reduction Achieved</td><td>{sec['total_reduction_achieved_pct']:.1f}%</td></tr>"
            f"<tr><td>Current Emissions</td><td>{sec['current_emissions_tco2e']:,.2f} tCO2e</td></tr>"
            f"<tr><td>SBTi Validated</td><td>{'Yes' if sec['sbti_validated'] else 'No'}</td></tr>"
            f"</table>"
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        """Render targets HTML."""
        sec = self._section_target_analysis(data)
        rows = ""
        for t in sec["targets"]:
            css = "on-track" if t["on_track"] else "delayed"
            rows += (
                f"<tr class='{css}'><td>{t['target_name']}</td>"
                f"<td>{t['target_reduction_pct']:.1f}%</td>"
                f"<td>{t['actual_reduction_pct']:.1f}%</td>"
                f"<td>{'Yes' if t['on_track'] else 'No'}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Target</th><th>Target %</th><th>Actual %</th>"
            f"<th>On Track</th></tr>{rows}</table>"
        )

    def _html_scopes(self, data: Dict[str, Any]) -> str:
        """Render scope breakdown HTML."""
        sec = self._section_scope_breakdown(data)
        rows = "".join(
            f"<tr><td>{s['scope']}</td><td>{s['base_year_tco2e']:,.2f}</td>"
            f"<td>{s['current_tco2e']:,.2f}</td>"
            f"<td>{s['reduction_pct']:.1f}%</td></tr>"
            for s in sec["scopes"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Scope</th><th>Base Year</th><th>Current</th>"
            f"<th>Reduction</th></tr>{rows}</table>"
        )

    def _html_paris_alignment(self, data: Dict[str, Any]) -> str:
        """Render Paris alignment HTML."""
        sec = self._section_paris_alignment(data)
        css = "aligned" if sec["aligned_with_2c"] else "not-aligned"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css}'>{sec['alignment_status']} "
            f"(Score: {sec['alignment_score_pct']:.1f}%)</p>\n"
            f"<p>Temperature Pathway: {sec['temperature_pathway']}</p>"
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        """Render milestones HTML."""
        sec = self._section_milestones(data)
        rows = ""
        for m in sec["milestones"]:
            css = "on-track" if m["status"] in ("completed", "on_track") else "delayed"
            rows += (
                f"<tr class='{css}'><td>{m['description'][:50]}</td>"
                f"<td>{m['target_date']}</td><td>{m['status']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Completion: {sec['completion_rate_pct']:.1f}%</p>\n"
            f"<table><tr><th>Milestone</th><th>Target Date</th><th>Status</th></tr>"
            f"{rows}</table>"
        )
