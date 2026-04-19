# -*- coding: utf-8 -*-
"""
PreventionMitigationReportTemplate - CSDDD Prevention and Mitigation Report

Renders prevention measures and mitigation actions with effectiveness tracking,
budget analysis, and gap identification per CSDDD Articles 8-9 requirements
for preventing potential adverse impacts and bringing actual impacts to an end.

Regulatory References:
    - Directive (EU) 2024/1760, Article 8 (Prevention of Potential Adverse Impacts)
    - Directive (EU) 2024/1760, Article 9 (Bringing Actual Adverse Impacts to an End)
    - Directive (EU) 2024/1760, Article 12 (Monitoring)
    - OECD Due Diligence Guidance for Responsible Business Conduct (2018)

Sections:
    1. Measure Overview - Summary of all prevention/mitigation measures
    2. Prevention Measures - Measures to prevent potential adverse impacts
    3. Mitigation Actions - Actions to address actual adverse impacts
    4. Effectiveness Assessment - KPI-based effectiveness tracking
    5. Budget Analysis - Financial resource allocation and spending
    6. Gap Analysis - Unaddressed impacts and resource gaps

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "measure_overview",
    "prevention_measures",
    "mitigation_actions",
    "effectiveness_assessment",
    "budget_analysis",
    "gap_analysis",
]

_MEASURE_STATUSES: List[str] = [
    "planned",
    "in_progress",
    "implemented",
    "verified",
    "closed",
]

_EFFECTIVENESS_RATINGS: List[str] = [
    "not_assessed",
    "ineffective",
    "partially_effective",
    "effective",
    "highly_effective",
]

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

def _effectiveness_score(rating: str) -> float:
    """Convert effectiveness rating to numeric score (0.0-1.0)."""
    mapping = {
        "not_assessed": 0.0,
        "ineffective": 0.1,
        "partially_effective": 0.5,
        "effective": 0.8,
        "highly_effective": 1.0,
    }
    return mapping.get(rating, 0.0)

class PreventionMitigationReportTemplate:
    """
    CSDDD Prevention and Mitigation Report.

    Renders a comprehensive overview of prevention measures for potential
    adverse impacts (Art 8) and mitigation actions for actual adverse
    impacts (Art 9), with effectiveness tracking based on KPIs, financial
    resource allocation analysis, and gap identification.

    Example:
        >>> tpl = PreventionMitigationReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PreventionMitigationReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
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
        if "prevention_measures" not in data and "mitigation_actions" not in data:
            errors.append(
                "At least one of prevention_measures or mitigation_actions is required"
            )
        if "budget" not in data:
            warnings.append("budget missing; budget section will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render prevention/mitigation report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_measure_overview(data),
            self._md_prevention_measures(data),
            self._md_mitigation_actions(data),
            self._md_effectiveness(data),
            self._md_budget(data),
            self._md_gap_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render prevention/mitigation report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_measure_overview(data),
            self._html_prevention(data),
            self._html_mitigation(data),
            self._html_effectiveness(data),
            self._html_budget(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Prevention & Mitigation Report - CSDDD</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render prevention/mitigation report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "prevention_mitigation_report",
            "directive_reference": "Directive (EU) 2024/1760, Art 8-9",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "measure_overview": self._section_measure_overview(data),
            "prevention_measures": self._section_prevention_measures(data),
            "mitigation_actions": self._section_mitigation_actions(data),
            "effectiveness_assessment": self._section_effectiveness_assessment(data),
            "budget_analysis": self._section_budget_analysis(data),
            "gap_analysis": self._section_gap_analysis(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_measure_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build measure overview section."""
        prevention = data.get("prevention_measures", [])
        mitigation = data.get("mitigation_actions", [])
        all_measures = prevention + mitigation
        status_counts: Dict[str, int] = {}
        for status in _MEASURE_STATUSES:
            status_counts[status] = sum(
                1 for m in all_measures if m.get("status") == status
            )
        total_budget = sum(m.get("budget_eur", 0.0) for m in all_measures)
        total_spent = sum(m.get("spent_eur", 0.0) for m in all_measures)
        effectiveness_scores = [
            _effectiveness_score(m.get("effectiveness", "not_assessed"))
            for m in all_measures
            if m.get("effectiveness") != "not_assessed"
        ]
        avg_effectiveness = (
            round(sum(effectiveness_scores) / len(effectiveness_scores) * 100, 1)
            if effectiveness_scores else 0.0
        )
        return {
            "title": "Measure Overview",
            "total_measures": len(all_measures),
            "prevention_count": len(prevention),
            "mitigation_count": len(mitigation),
            "status_distribution": status_counts,
            "total_budget_eur": round(total_budget, 2),
            "total_spent_eur": round(total_spent, 2),
            "budget_utilisation_pct": (
                round(total_spent / total_budget * 100, 1) if total_budget > 0 else 0.0
            ),
            "average_effectiveness_pct": avg_effectiveness,
        }

    def _section_prevention_measures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build prevention measures section (Art 8)."""
        measures = data.get("prevention_measures", [])
        entries: List[Dict[str, Any]] = []
        for m in measures:
            entries.append({
                "measure_id": m.get("measure_id", ""),
                "title": m.get("title", ""),
                "description": m.get("description", ""),
                "target_impact": m.get("target_impact", ""),
                "domain": m.get("domain", ""),
                "status": m.get("status", "planned"),
                "start_date": m.get("start_date", ""),
                "target_date": m.get("target_date", ""),
                "responsible": m.get("responsible", ""),
                "budget_eur": round(m.get("budget_eur", 0.0), 2),
                "spent_eur": round(m.get("spent_eur", 0.0), 2),
                "effectiveness": m.get("effectiveness", "not_assessed"),
                "kpis": m.get("kpis", []),
                "value_chain_scope": m.get("value_chain_scope", ""),
                "contractual_assurance": m.get("contractual_assurance", False),
            })
        by_domain: Dict[str, int] = {}
        for e in entries:
            d = e["domain"]
            by_domain[d] = by_domain.get(d, 0) + 1
        return {
            "title": "Prevention Measures (Art 8)",
            "total_prevention_measures": len(entries),
            "measures": entries,
            "domain_distribution": by_domain,
            "measures_with_contractual": sum(
                1 for e in entries if e["contractual_assurance"]
            ),
        }

    def _section_mitigation_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build mitigation actions section (Art 9)."""
        actions = data.get("mitigation_actions", [])
        entries: List[Dict[str, Any]] = []
        for a in actions:
            entries.append({
                "action_id": a.get("action_id", ""),
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "adverse_impact_ref": a.get("adverse_impact_ref", ""),
                "domain": a.get("domain", ""),
                "status": a.get("status", "planned"),
                "severity_of_impact": a.get("severity_of_impact", ""),
                "remediation_type": a.get("remediation_type", ""),
                "start_date": a.get("start_date", ""),
                "target_date": a.get("target_date", ""),
                "responsible": a.get("responsible", ""),
                "budget_eur": round(a.get("budget_eur", 0.0), 2),
                "spent_eur": round(a.get("spent_eur", 0.0), 2),
                "effectiveness": a.get("effectiveness", "not_assessed"),
                "kpis": a.get("kpis", []),
                "stakeholders_consulted": a.get("stakeholders_consulted", []),
                "includes_remediation": a.get("includes_remediation", False),
            })
        by_remediation: Dict[str, int] = {}
        for e in entries:
            rt = e["remediation_type"] or "unspecified"
            by_remediation[rt] = by_remediation.get(rt, 0) + 1
        return {
            "title": "Mitigation Actions (Art 9)",
            "total_mitigation_actions": len(entries),
            "actions": entries,
            "remediation_type_distribution": by_remediation,
            "actions_with_remediation": sum(
                1 for e in entries if e["includes_remediation"]
            ),
        }

    def _section_effectiveness_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build effectiveness assessment section."""
        prevention = data.get("prevention_measures", [])
        mitigation = data.get("mitigation_actions", [])
        all_measures = prevention + mitigation
        rating_counts: Dict[str, int] = {}
        for rating in _EFFECTIVENESS_RATINGS:
            rating_counts[rating] = sum(
                1 for m in all_measures if m.get("effectiveness") == rating
            )
        kpi_results: List[Dict[str, Any]] = []
        for m in all_measures:
            for kpi in m.get("kpis", []):
                kpi_results.append({
                    "measure_id": m.get("measure_id", m.get("action_id", "")),
                    "measure_title": m.get("title", ""),
                    "kpi_name": kpi.get("name", ""),
                    "target_value": kpi.get("target_value", 0),
                    "actual_value": kpi.get("actual_value", 0),
                    "unit": kpi.get("unit", ""),
                    "on_target": kpi.get("actual_value", 0) >= kpi.get("target_value", 0),
                })
        kpis_on_target = sum(1 for k in kpi_results if k["on_target"])
        return {
            "title": "Effectiveness Assessment",
            "total_measures_assessed": len(all_measures),
            "rating_distribution": rating_counts,
            "overall_effectiveness_pct": self._calc_overall_effectiveness(all_measures),
            "kpi_results": kpi_results,
            "total_kpis": len(kpi_results),
            "kpis_on_target": kpis_on_target,
            "kpi_achievement_pct": (
                round(kpis_on_target / len(kpi_results) * 100, 1)
                if kpi_results else 0.0
            ),
        }

    def _section_budget_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build budget analysis section."""
        budget = data.get("budget", {})
        prevention = data.get("prevention_measures", [])
        mitigation = data.get("mitigation_actions", [])
        prevention_budget = sum(m.get("budget_eur", 0.0) for m in prevention)
        prevention_spent = sum(m.get("spent_eur", 0.0) for m in prevention)
        mitigation_budget = sum(a.get("budget_eur", 0.0) for a in mitigation)
        mitigation_spent = sum(a.get("spent_eur", 0.0) for a in mitigation)
        total_budget = prevention_budget + mitigation_budget
        total_spent = prevention_spent + mitigation_spent
        return {
            "title": "Budget Analysis",
            "total_budget_eur": round(budget.get("total_allocated_eur", total_budget), 2),
            "total_spent_eur": round(total_spent, 2),
            "remaining_eur": round(
                budget.get("total_allocated_eur", total_budget) - total_spent, 2
            ),
            "utilisation_pct": round(
                total_spent / budget.get("total_allocated_eur", total_budget) * 100, 1
            ) if budget.get("total_allocated_eur", total_budget) > 0 else 0.0,
            "prevention_budget_eur": round(prevention_budget, 2),
            "prevention_spent_eur": round(prevention_spent, 2),
            "mitigation_budget_eur": round(mitigation_budget, 2),
            "mitigation_spent_eur": round(mitigation_spent, 2),
            "fte_allocated": budget.get("fte_allocated", 0),
            "external_consultants": budget.get("external_consultants", 0),
            "year_over_year_change_pct": round(
                budget.get("yoy_change_pct", 0.0), 1
            ),
        }

    def _section_gap_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap analysis section."""
        gaps = data.get("pm_gaps", [])
        prevention = data.get("prevention_measures", [])
        mitigation = data.get("mitigation_actions", [])
        unaddressed_impacts = data.get("unaddressed_impacts", [])
        stalled_measures = [
            m for m in prevention + mitigation
            if m.get("status") in ("planned",) and m.get("overdue", False)
        ]
        ineffective_measures = [
            m for m in prevention + mitigation
            if m.get("effectiveness") == "ineffective"
        ]
        return {
            "title": "Gap Analysis",
            "total_gaps": len(gaps) + len(unaddressed_impacts) + len(stalled_measures),
            "unaddressed_impacts": [
                {
                    "impact_id": u.get("impact_id", ""),
                    "description": u.get("description", ""),
                    "domain": u.get("domain", ""),
                    "severity": u.get("severity", ""),
                }
                for u in unaddressed_impacts
            ],
            "stalled_measures": [
                {
                    "measure_id": m.get("measure_id", m.get("action_id", "")),
                    "title": m.get("title", ""),
                    "status": m.get("status", ""),
                    "target_date": m.get("target_date", ""),
                }
                for m in stalled_measures
            ],
            "ineffective_measures": [
                {
                    "measure_id": m.get("measure_id", m.get("action_id", "")),
                    "title": m.get("title", ""),
                    "effectiveness": m.get("effectiveness", ""),
                }
                for m in ineffective_measures
            ],
            "resource_gaps": gaps,
            "total_unaddressed": len(unaddressed_impacts),
            "total_stalled": len(stalled_measures),
            "total_ineffective": len(ineffective_measures),
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _calc_overall_effectiveness(
        self, measures: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall effectiveness percentage."""
        assessed = [
            m for m in measures
            if m.get("effectiveness") not in ("not_assessed", None, "")
        ]
        if not assessed:
            return 0.0
        scores = [_effectiveness_score(m.get("effectiveness", "")) for m in assessed]
        return round(sum(scores) / len(scores) * 100, 1)

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Prevention & Mitigation Report - CSDDD\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 8-9"
        )

    def _md_measure_overview(self, data: Dict[str, Any]) -> str:
        """Render measure overview as markdown."""
        sec = self._section_measure_overview(data)
        lines = [
            f"## {sec['title']}\n",
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Measures | {sec['total_measures']} |\n"
            f"| Prevention Measures | {sec['prevention_count']} |\n"
            f"| Mitigation Actions | {sec['mitigation_count']} |\n"
            f"| Total Budget (EUR) | {sec['total_budget_eur']:,.2f} |\n"
            f"| Total Spent (EUR) | {sec['total_spent_eur']:,.2f} |\n"
            f"| Budget Utilisation | {sec['budget_utilisation_pct']:.1f}% |\n"
            f"| Avg Effectiveness | {sec['average_effectiveness_pct']:.1f}% |",
        ]
        if sec["status_distribution"]:
            lines.append("\n**Status Distribution:**")
            for status, count in sec["status_distribution"].items():
                lines.append(f"- {status.replace('_', ' ').title()}: {count}")
        return "\n".join(lines)

    def _md_prevention_measures(self, data: Dict[str, Any]) -> str:
        """Render prevention measures as markdown."""
        sec = self._section_prevention_measures(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Prevention Measures:** {sec['total_prevention_measures']}  \n"
            f"**With Contractual Assurance:** {sec['measures_with_contractual']}\n",
            "| Measure | Domain | Status | Budget (EUR) | Effectiveness |",
            "|---------|--------|--------|------------:|---------------|",
        ]
        for m in sec["measures"]:
            lines.append(
                f"| {m['title'][:40]} | {m['domain']} | "
                f"{m['status']} | {m['budget_eur']:,.2f} | "
                f"{m['effectiveness'].replace('_', ' ').title()} |"
            )
        return "\n".join(lines)

    def _md_mitigation_actions(self, data: Dict[str, Any]) -> str:
        """Render mitigation actions as markdown."""
        sec = self._section_mitigation_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Mitigation Actions:** {sec['total_mitigation_actions']}  \n"
            f"**With Remediation:** {sec['actions_with_remediation']}\n",
            "| Action | Domain | Severity | Status | Budget (EUR) |",
            "|--------|--------|----------|--------|------------:|",
        ]
        for a in sec["actions"]:
            lines.append(
                f"| {a['title'][:40]} | {a['domain']} | "
                f"{a['severity_of_impact']} | {a['status']} | "
                f"{a['budget_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_effectiveness(self, data: Dict[str, Any]) -> str:
        """Render effectiveness assessment as markdown."""
        sec = self._section_effectiveness_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall Effectiveness:** {sec['overall_effectiveness_pct']:.1f}%  \n"
            f"**KPIs On Target:** {sec['kpis_on_target']} / {sec['total_kpis']} "
            f"({sec['kpi_achievement_pct']:.1f}%)\n",
            "**Rating Distribution:**",
        ]
        for rating, count in sec["rating_distribution"].items():
            lines.append(f"- {rating.replace('_', ' ').title()}: {count}")
        if sec["kpi_results"]:
            lines.append("\n| Measure | KPI | Target | Actual | On Target |")
            lines.append("|---------|-----|-------:|-------:|:---------:|")
            for k in sec["kpi_results"][:15]:
                on_target = "Yes" if k["on_target"] else "No"
                lines.append(
                    f"| {k['measure_title'][:30]} | {k['kpi_name']} | "
                    f"{k['target_value']} | {k['actual_value']} | {on_target} |"
                )
        return "\n".join(lines)

    def _md_budget(self, data: Dict[str, Any]) -> str:
        """Render budget analysis as markdown."""
        sec = self._section_budget_analysis(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Category | Budget (EUR) | Spent (EUR) |\n"
            f"|----------|------------:|----------:|\n"
            f"| Prevention | {sec['prevention_budget_eur']:,.2f} | "
            f"{sec['prevention_spent_eur']:,.2f} |\n"
            f"| Mitigation | {sec['mitigation_budget_eur']:,.2f} | "
            f"{sec['mitigation_spent_eur']:,.2f} |\n"
            f"| **Total** | **{sec['total_budget_eur']:,.2f}** | "
            f"**{sec['total_spent_eur']:,.2f}** |\n\n"
            f"**Utilisation:** {sec['utilisation_pct']:.1f}%  \n"
            f"**Remaining:** EUR {sec['remaining_eur']:,.2f}  \n"
            f"**FTE Allocated:** {sec['fte_allocated']}  \n"
            f"**YoY Change:** {sec['year_over_year_change_pct']:.1f}%"
        )

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis as markdown."""
        sec = self._section_gap_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps:** {sec['total_gaps']}  \n"
            f"**Unaddressed Impacts:** {sec['total_unaddressed']}  \n"
            f"**Stalled Measures:** {sec['total_stalled']}  \n"
            f"**Ineffective Measures:** {sec['total_ineffective']}\n",
        ]
        if sec["unaddressed_impacts"]:
            lines.append("### Unaddressed Impacts")
            for u in sec["unaddressed_impacts"]:
                lines.append(
                    f"- **{u['domain']}** ({u['severity']}): {u['description']}"
                )
        if sec["ineffective_measures"]:
            lines.append("\n### Ineffective Measures")
            for m in sec["ineffective_measures"]:
                lines.append(f"- {m['title']} (ID: {m['measure_id']})")
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
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".effective{color:#2e7d32;font-weight:bold}"
            ".ineffective{color:#c62828;font-weight:bold}"
            ".partial{color:#ef6c00}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Prevention & Mitigation Report - CSDDD</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_measure_overview(self, data: Dict[str, Any]) -> str:
        """Render measure overview HTML."""
        sec = self._section_measure_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Total Measures</td><td>{sec['total_measures']}</td></tr>"
            f"<tr><td>Prevention</td><td>{sec['prevention_count']}</td></tr>"
            f"<tr><td>Mitigation</td><td>{sec['mitigation_count']}</td></tr>"
            f"<tr><td>Budget (EUR)</td><td>{sec['total_budget_eur']:,.2f}</td></tr>"
            f"<tr><td>Effectiveness</td><td>{sec['average_effectiveness_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_prevention(self, data: Dict[str, Any]) -> str:
        """Render prevention measures HTML."""
        sec = self._section_prevention_measures(data)
        rows = "".join(
            f"<tr><td>{m['title'][:50]}</td><td>{m['domain']}</td>"
            f"<td>{m['status']}</td><td>{m['budget_eur']:,.2f}</td></tr>"
            for m in sec["measures"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_prevention_measures']}</p>\n"
            f"<table><tr><th>Measure</th><th>Domain</th><th>Status</th>"
            f"<th>Budget (EUR)</th></tr>{rows}</table>"
        )

    def _html_mitigation(self, data: Dict[str, Any]) -> str:
        """Render mitigation actions HTML."""
        sec = self._section_mitigation_actions(data)
        rows = "".join(
            f"<tr><td>{a['title'][:50]}</td><td>{a['domain']}</td>"
            f"<td>{a['status']}</td><td>{a['budget_eur']:,.2f}</td></tr>"
            for a in sec["actions"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_mitigation_actions']}</p>\n"
            f"<table><tr><th>Action</th><th>Domain</th><th>Status</th>"
            f"<th>Budget (EUR)</th></tr>{rows}</table>"
        )

    def _html_effectiveness(self, data: Dict[str, Any]) -> str:
        """Render effectiveness assessment HTML."""
        sec = self._section_effectiveness_assessment(data)
        eff_class = "effective" if sec["overall_effectiveness_pct"] >= 70.0 else "partial"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{eff_class}'>Overall Effectiveness: "
            f"{sec['overall_effectiveness_pct']:.1f}%</p>\n"
            f"<p>KPIs On Target: {sec['kpis_on_target']} / {sec['total_kpis']}</p>"
        )

    def _html_budget(self, data: Dict[str, Any]) -> str:
        """Render budget analysis HTML."""
        sec = self._section_budget_analysis(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>Budget</th><th>Spent</th></tr>"
            f"<tr><td>Prevention</td><td>EUR {sec['prevention_budget_eur']:,.2f}</td>"
            f"<td>EUR {sec['prevention_spent_eur']:,.2f}</td></tr>"
            f"<tr><td>Mitigation</td><td>EUR {sec['mitigation_budget_eur']:,.2f}</td>"
            f"<td>EUR {sec['mitigation_spent_eur']:,.2f}</td></tr>"
            f"<tr><td><strong>Total</strong></td>"
            f"<td><strong>EUR {sec['total_budget_eur']:,.2f}</strong></td>"
            f"<td><strong>EUR {sec['total_spent_eur']:,.2f}</strong></td></tr>"
            f"</table>\n"
            f"<p>Utilisation: {sec['utilisation_pct']:.1f}%</p>"
        )
