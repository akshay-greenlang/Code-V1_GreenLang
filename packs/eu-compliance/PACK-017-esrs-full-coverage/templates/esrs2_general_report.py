# -*- coding: utf-8 -*-
"""
ESRS2GeneralReportTemplate - ESRS 2 General Disclosures Report

Renders governance structures, board composition, sustainability committees,
incentive schemes, due diligence processes, risk management, strategy,
stakeholder engagement, material IROs, and disclosure coverage per ESRS 2.

Sections:
    1. Governance Overview
    2. Board Composition
    3. Sustainability Committees
    4. Incentive Schemes
    5. Due Diligence
    6. Risk Management
    7. Strategy Overview
    8. Stakeholder Engagement
    9. Material IROs
    10. Disclosure Coverage

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "governance_overview",
    "board_composition",
    "sustainability_committees",
    "incentive_schemes",
    "due_diligence",
    "risk_management",
    "strategy_overview",
    "stakeholder_engagement",
    "material_iros",
    "disclosure_coverage",
]

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

class ESRS2GeneralReportTemplate:
    """
    ESRS 2 General Disclosures report template.

    Renders governance, strategy, due diligence, risk management,
    stakeholder engagement, material impacts/risks/opportunities, and
    overall disclosure coverage mapping per ESRS 2.

    Example:
        >>> tpl = ESRS2GeneralReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRS2GeneralReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
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
        if "board_members" not in data:
            warnings.append("board_members missing; will default to empty")
        if "material_iros" not in data:
            warnings.append("material_iros missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS 2 General Disclosures report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_governance_overview(data),
            self._md_board_composition(data),
            self._md_sustainability_committees(data),
            self._md_incentive_schemes(data),
            self._md_due_diligence(data),
            self._md_risk_management(data),
            self._md_strategy_overview(data),
            self._md_stakeholder_engagement(data),
            self._md_material_iros(data),
            self._md_disclosure_coverage(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS 2 General Disclosures report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_governance_overview(data),
            self._html_board_composition(data),
            self._html_material_iros(data),
            self._html_disclosure_coverage(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS 2 General Disclosures Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render ESRS 2 General Disclosures report as JSON string."""
        self.generated_at = utcnow()
        result = {
            "template": "esrs2_general_report",
            "esrs_reference": "ESRS 2",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "board_size": len(data.get("board_members", [])),
            "committee_count": len(data.get("sustainability_committees", [])),
            "material_iro_count": len(data.get("material_iros", [])),
            "disclosure_standards_covered": data.get("disclosure_standards_covered", []),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_governance_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build governance overview section."""
        return {
            "title": "Governance Overview",
            "governance_model": data.get("governance_model", ""),
            "sustainability_governance_description": data.get(
                "sustainability_governance_description", ""
            ),
            "admin_management_body": data.get("admin_management_body", ""),
            "reporting_lines": data.get("reporting_lines", []),
            "frequency_of_review": data.get("frequency_of_review", ""),
        }

    def _section_board_composition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build board composition section."""
        members = data.get("board_members", [])
        total = len(members)
        independent = sum(1 for m in members if m.get("independent", False))
        return {
            "title": "Board Composition",
            "total_members": total,
            "independent_members": independent,
            "independence_ratio": round(independent / total * 100, 1) if total > 0 else 0.0,
            "gender_diversity": data.get("board_gender_diversity", {}),
            "sustainability_expertise_count": sum(
                1 for m in members if m.get("sustainability_expertise", False)
            ),
            "members": [
                {
                    "name": m.get("name", ""),
                    "role": m.get("role", ""),
                    "independent": m.get("independent", False),
                    "sustainability_expertise": m.get("sustainability_expertise", False),
                }
                for m in members
            ],
        }

    def _section_sustainability_committees(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sustainability committees section."""
        committees = data.get("sustainability_committees", [])
        return {
            "title": "Sustainability Committees",
            "count": len(committees),
            "committees": [
                {
                    "name": c.get("name", ""),
                    "mandate": c.get("mandate", ""),
                    "members_count": c.get("members_count", 0),
                    "meeting_frequency": c.get("meeting_frequency", ""),
                    "chair": c.get("chair", ""),
                }
                for c in committees
            ],
        }

    def _section_incentive_schemes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build incentive schemes section."""
        schemes = data.get("incentive_schemes", [])
        return {
            "title": "Incentive Schemes Linked to Sustainability",
            "count": len(schemes),
            "schemes": [
                {
                    "scheme_name": s.get("scheme_name", ""),
                    "linked_targets": s.get("linked_targets", []),
                    "beneficiary_group": s.get("beneficiary_group", ""),
                    "weight_percentage": s.get("weight_percentage", 0.0),
                }
                for s in schemes
            ],
        }

    def _section_due_diligence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build due diligence section."""
        return {
            "title": "Due Diligence",
            "due_diligence_description": data.get("due_diligence_description", ""),
            "process_stages": data.get("due_diligence_stages", []),
            "value_chain_coverage": data.get("value_chain_due_diligence_coverage", ""),
            "human_rights_dd": data.get("human_rights_due_diligence", False),
            "environmental_dd": data.get("environmental_due_diligence", False),
        }

    def _section_risk_management(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk management section."""
        risks = data.get("sustainability_risks", [])
        return {
            "title": "Risk Management",
            "integration_with_erm": data.get("integration_with_erm", False),
            "risk_identification_process": data.get("risk_identification_process", ""),
            "risk_count": len(risks),
            "risks": [
                {
                    "risk_name": r.get("risk_name", ""),
                    "category": r.get("category", ""),
                    "likelihood": r.get("likelihood", ""),
                    "impact": r.get("impact", ""),
                    "mitigation": r.get("mitigation", ""),
                }
                for r in risks
            ],
        }

    def _section_strategy_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build strategy overview section."""
        return {
            "title": "Strategy Overview",
            "business_model_description": data.get("business_model_description", ""),
            "sustainability_strategy": data.get("sustainability_strategy", ""),
            "value_chain_description": data.get("value_chain_description", ""),
            "key_stakeholders": data.get("key_stakeholders", []),
            "time_horizons": data.get("time_horizons", {}),
        }

    def _section_stakeholder_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build stakeholder engagement section."""
        groups = data.get("stakeholder_groups", [])
        return {
            "title": "Stakeholder Engagement",
            "engagement_policy": data.get("engagement_policy", ""),
            "group_count": len(groups),
            "groups": [
                {
                    "name": g.get("name", ""),
                    "engagement_method": g.get("engagement_method", ""),
                    "frequency": g.get("frequency", ""),
                    "key_topics": g.get("key_topics", []),
                }
                for g in groups
            ],
        }

    def _section_material_iros(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build material impacts, risks and opportunities section."""
        iros = data.get("material_iros", [])
        return {
            "title": "Material Impacts, Risks and Opportunities",
            "materiality_assessment_method": data.get("materiality_assessment_method", ""),
            "double_materiality_applied": data.get("double_materiality_applied", False),
            "total_iros": len(iros),
            "iros": [
                {
                    "topic": i.get("topic", ""),
                    "type": i.get("type", ""),
                    "impact_materiality": i.get("impact_materiality", ""),
                    "financial_materiality": i.get("financial_materiality", ""),
                    "esrs_standard_mapped": i.get("esrs_standard_mapped", ""),
                }
                for i in iros
            ],
        }

    def _section_disclosure_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build disclosure coverage section."""
        standards = data.get("disclosure_coverage", [])
        total = len(standards)
        covered = sum(1 for s in standards if s.get("covered", False))
        return {
            "title": "Disclosure Coverage",
            "total_standards": total,
            "covered_standards": covered,
            "coverage_percentage": round(covered / total * 100, 1) if total > 0 else 0.0,
            "standards": [
                {
                    "standard": s.get("standard", ""),
                    "covered": s.get("covered", False),
                    "disclosure_requirements_met": s.get("disclosure_requirements_met", 0),
                    "disclosure_requirements_total": s.get("disclosure_requirements_total", 0),
                    "phase_in_applicable": s.get("phase_in_applicable", False),
                }
                for s in standards
            ],
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
            f"# ESRS 2 General Disclosures Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS 2 General Disclosures"
        )

    def _md_governance_overview(self, data: Dict[str, Any]) -> str:
        """Render governance overview markdown."""
        sec = self._section_governance_overview(data)
        lines_list = sec['reporting_lines'] if sec['reporting_lines'] else []
        return (
            f"## {sec['title']}\n\n"
            f"**Governance Model:** {sec['governance_model']}  \n"
            f"**Reporting Lines:** {', '.join(lines_list) if lines_list else 'N/A'}  \n"
            f"**Review Frequency:** {sec['frequency_of_review']}\n\n"
            f"{sec['sustainability_governance_description']}"
        )

    def _md_board_composition(self, data: Dict[str, Any]) -> str:
        """Render board composition markdown."""
        sec = self._section_board_composition(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Members:** {sec['total_members']}  ",
            f"**Independent:** {sec['independent_members']} ({sec['independence_ratio']:.1f}%)  ",
            f"**Sustainability Expertise:** {sec['sustainability_expertise_count']}\n",
        ]
        if sec["members"]:
            lines.append("| Name | Role | Independent | Sustainability Expertise |")
            lines.append("|------|------|:-----------:|:------------------------:|")
            for m in sec["members"]:
                ind = "Yes" if m["independent"] else "No"
                exp = "Yes" if m["sustainability_expertise"] else "No"
                lines.append(f"| {m['name']} | {m['role']} | {ind} | {exp} |")
        return "\n".join(lines)

    def _md_sustainability_committees(self, data: Dict[str, Any]) -> str:
        """Render sustainability committees markdown."""
        sec = self._section_sustainability_committees(data)
        lines = [f"## {sec['title']}\n"]
        for c in sec["committees"]:
            lines.append(f"### {c['name']}")
            lines.append(f"- **Mandate:** {c['mandate']}")
            lines.append(f"- **Members:** {c['members_count']}")
            lines.append(f"- **Meeting Frequency:** {c['meeting_frequency']}")
            lines.append(f"- **Chair:** {c['chair']}\n")
        return "\n".join(lines)

    def _md_incentive_schemes(self, data: Dict[str, Any]) -> str:
        """Render incentive schemes markdown."""
        sec = self._section_incentive_schemes(data)
        lines = [f"## {sec['title']}\n"]
        if sec["schemes"]:
            lines.append("| Scheme | Beneficiary | Weight | Linked Targets |")
            lines.append("|--------|-------------|-------:|----------------|")
            for s in sec["schemes"]:
                targets = ", ".join(s["linked_targets"])
                lines.append(
                    f"| {s['scheme_name']} | {s['beneficiary_group']} "
                    f"| {s['weight_percentage']:.1f}% | {targets} |"
                )
        return "\n".join(lines)

    def _md_due_diligence(self, data: Dict[str, Any]) -> str:
        """Render due diligence markdown."""
        sec = self._section_due_diligence(data)
        hr = "Yes" if sec["human_rights_dd"] else "No"
        env = "Yes" if sec["environmental_dd"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**Human Rights DD:** {hr}  \n"
            f"**Environmental DD:** {env}  \n"
            f"**Value Chain Coverage:** {sec['value_chain_coverage']}\n\n"
            f"{sec['due_diligence_description']}"
        )

    def _md_risk_management(self, data: Dict[str, Any]) -> str:
        """Render risk management markdown."""
        sec = self._section_risk_management(data)
        erm = "Yes" if sec["integration_with_erm"] else "No"
        lines = [
            f"## {sec['title']}\n",
            f"**Integrated with ERM:** {erm}  ",
            f"**Risks Identified:** {sec['risk_count']}\n",
        ]
        if sec["risks"]:
            lines.append("| Risk | Category | Likelihood | Impact |")
            lines.append("|------|----------|------------|--------|")
            for r in sec["risks"]:
                lines.append(
                    f"| {r['risk_name']} | {r['category']} "
                    f"| {r['likelihood']} | {r['impact']} |"
                )
        return "\n".join(lines)

    def _md_strategy_overview(self, data: Dict[str, Any]) -> str:
        """Render strategy overview markdown."""
        sec = self._section_strategy_overview(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Business Model:** {sec['business_model_description']}\n\n"
            f"**Sustainability Strategy:** {sec['sustainability_strategy']}\n\n"
            f"**Value Chain:** {sec['value_chain_description']}"
        )

    def _md_stakeholder_engagement(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement markdown."""
        sec = self._section_stakeholder_engagement(data)
        lines = [f"## {sec['title']}\n", f"**Policy:** {sec['engagement_policy']}\n"]
        if sec["groups"]:
            lines.append("| Stakeholder Group | Method | Frequency |")
            lines.append("|-------------------|--------|-----------|")
            for g in sec["groups"]:
                lines.append(
                    f"| {g['name']} | {g['engagement_method']} | {g['frequency']} |"
                )
        return "\n".join(lines)

    def _md_material_iros(self, data: Dict[str, Any]) -> str:
        """Render material IROs markdown."""
        sec = self._section_material_iros(data)
        dm = "Yes" if sec["double_materiality_applied"] else "No"
        lines = [
            f"## {sec['title']}\n",
            f"**Double Materiality Applied:** {dm}  ",
            f"**Total IROs:** {sec['total_iros']}\n",
        ]
        if sec["iros"]:
            lines.append("| Topic | Type | Impact | Financial | ESRS Standard |")
            lines.append("|-------|------|--------|-----------|---------------|")
            for i in sec["iros"]:
                lines.append(
                    f"| {i['topic']} | {i['type']} | {i['impact_materiality']} "
                    f"| {i['financial_materiality']} | {i['esrs_standard_mapped']} |"
                )
        return "\n".join(lines)

    def _md_disclosure_coverage(self, data: Dict[str, Any]) -> str:
        """Render disclosure coverage markdown."""
        sec = self._section_disclosure_coverage(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Coverage:** {sec['covered_standards']}/{sec['total_standards']} "
            f"({sec['coverage_percentage']:.1f}%)\n",
        ]
        if sec["standards"]:
            lines.append("| Standard | Covered | DRs Met | DRs Total | Phase-In |")
            lines.append("|----------|:-------:|--------:|----------:|:--------:|")
            for s in sec["standards"]:
                cov = "Yes" if s["covered"] else "No"
                ph = "Yes" if s["phase_in_applicable"] else "No"
                lines.append(
                    f"| {s['standard']} | {cov} | {s['disclosure_requirements_met']} "
                    f"| {s['disclosure_requirements_total']} | {ph} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

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
            ".total{font-weight:bold;background:#e8f5e9}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>ESRS 2 General Disclosures Report</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_governance_overview(self, data: Dict[str, Any]) -> str:
        """Render governance overview HTML."""
        sec = self._section_governance_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Model:</strong> {sec['governance_model']}</p>\n"
            f"<p>{sec['sustainability_governance_description']}</p>"
        )

    def _html_board_composition(self, data: Dict[str, Any]) -> str:
        """Render board composition HTML."""
        sec = self._section_board_composition(data)
        rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['role']}</td>"
            f"<td>{'Yes' if m['independent'] else 'No'}</td></tr>"
            for m in sec["members"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_members']} | "
            f"Independent: {sec['independent_members']}</p>\n"
            f"<table><tr><th>Name</th><th>Role</th><th>Independent</th></tr>"
            f"{rows}</table>"
        )

    def _html_material_iros(self, data: Dict[str, Any]) -> str:
        """Render material IROs HTML."""
        sec = self._section_material_iros(data)
        rows = "".join(
            f"<tr><td>{i['topic']}</td><td>{i['type']}</td>"
            f"<td>{i['esrs_standard_mapped']}</td></tr>"
            for i in sec["iros"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total IROs: {sec['total_iros']}</p>\n"
            f"<table><tr><th>Topic</th><th>Type</th><th>ESRS Standard</th></tr>"
            f"{rows}</table>"
        )

    def _html_disclosure_coverage(self, data: Dict[str, Any]) -> str:
        """Render disclosure coverage HTML."""
        sec = self._section_disclosure_coverage(data)
        rows = "".join(
            f"<tr><td>{s['standard']}</td>"
            f"<td>{'Yes' if s['covered'] else 'No'}</td>"
            f"<td>{s['disclosure_requirements_met']}/{s['disclosure_requirements_total']}</td></tr>"
            for s in sec["standards"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Coverage: {sec['coverage_percentage']:.1f}%</p>\n"
            f"<table><tr><th>Standard</th><th>Covered</th><th>DRs</th></tr>"
            f"{rows}</table>"
        )
