# -*- coding: utf-8 -*-
"""
ImpactAssessmentReportTemplate - CSDDD Adverse Impact Assessment Report

Renders a detailed adverse impact identification report with severity/likelihood
risk matrix, human rights and environmental impact categorisation, impact
prioritisation per CSDDD Art 6-7, and stakeholder impact analysis.

Regulatory References:
    - Directive (EU) 2024/1760, Article 6 (Identifying and Assessing Impacts)
    - Directive (EU) 2024/1760, Article 7 (Prioritisation of Identified Impacts)
    - UN Guiding Principles on Business and Human Rights (2011)
    - OECD Guidelines for Multinational Enterprises (2023 revision)
    - ILO Declaration on Fundamental Principles and Rights at Work

Sections:
    1. Impact Summary - Aggregate view of identified adverse impacts
    2. Human Rights Impacts - Human rights impact catalogue
    3. Environmental Impacts - Environmental impact catalogue
    4. Risk Matrix - Severity x Likelihood scoring grid
    5. Prioritisation - Ranked impact prioritisation per Art 7
    6. Stakeholder Impacts - Impact on affected stakeholders

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
    "impact_summary",
    "human_rights_impacts",
    "environmental_impacts",
    "risk_matrix",
    "prioritisation",
    "stakeholder_impacts",
]

# Standard human rights impact categories under CSDDD
_HR_CATEGORIES: List[str] = [
    "forced_labour",
    "child_labour",
    "workplace_health_safety",
    "freedom_of_association",
    "collective_bargaining",
    "discrimination",
    "adequate_living_wage",
    "working_hours",
    "privacy_and_data_protection",
    "land_and_housing_rights",
    "indigenous_peoples_rights",
    "right_to_water",
    "right_to_food",
    "freedom_of_expression",
]

# Standard environmental impact categories under CSDDD
_ENV_CATEGORIES: List[str] = [
    "climate_change_ghg",
    "air_pollution",
    "water_pollution",
    "soil_contamination",
    "biodiversity_loss",
    "deforestation",
    "hazardous_waste",
    "water_consumption",
    "resource_depletion",
    "ocean_degradation",
]

_SEVERITY_LEVELS: List[str] = ["negligible", "minor", "moderate", "major", "critical"]
_LIKELIHOOD_LEVELS: List[str] = ["rare", "unlikely", "possible", "likely", "almost_certain"]

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

def _severity_score(level: str) -> int:
    """Convert severity level to numeric score (1-5)."""
    mapping = {v: i + 1 for i, v in enumerate(_SEVERITY_LEVELS)}
    return mapping.get(level, 0)

def _likelihood_score(level: str) -> int:
    """Convert likelihood level to numeric score (1-5)."""
    mapping = {v: i + 1 for i, v in enumerate(_LIKELIHOOD_LEVELS)}
    return mapping.get(level, 0)

def _compute_risk_score(severity: str, likelihood: str) -> float:
    """Compute risk score as product of severity and likelihood (1-25)."""
    return float(_severity_score(severity) * _likelihood_score(likelihood))

def _risk_rating(score: float) -> str:
    """Map numeric risk score to a risk rating label."""
    if score >= 20:
        return "critical"
    elif score >= 12:
        return "high"
    elif score >= 6:
        return "medium"
    elif score >= 2:
        return "low"
    else:
        return "negligible"

class ImpactAssessmentReportTemplate:
    """
    CSDDD Adverse Impact Assessment Report.

    Renders identified adverse impacts across human rights and environmental
    dimensions with a severity/likelihood risk matrix, CSDDD Art 7
    prioritisation scoring, and stakeholder impact analysis. Follows
    deterministic scoring - no LLM calls for risk calculations.

    Example:
        >>> tpl = ImpactAssessmentReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImpactAssessmentReportTemplate."""
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
        if "impacts" not in data:
            errors.append("impacts list is required")
        if "stakeholders" not in data:
            warnings.append("stakeholders missing; stakeholder section will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render impact assessment report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_impact_summary(data),
            self._md_human_rights(data),
            self._md_environmental(data),
            self._md_risk_matrix(data),
            self._md_prioritisation(data),
            self._md_stakeholder_impacts(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render impact assessment report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_impact_summary(data),
            self._html_human_rights(data),
            self._html_environmental(data),
            self._html_risk_matrix(data),
            self._html_prioritisation(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Adverse Impact Assessment - CSDDD</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render impact assessment report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "impact_assessment_report",
            "directive_reference": "Directive (EU) 2024/1760, Art 6-7",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "impact_summary": self._section_impact_summary(data),
            "human_rights_impacts": self._section_human_rights_impacts(data),
            "environmental_impacts": self._section_environmental_impacts(data),
            "risk_matrix": self._section_risk_matrix(data),
            "prioritisation": self._section_prioritisation(data),
            "stakeholder_impacts": self._section_stakeholder_impacts(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_impact_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build impact summary section."""
        impacts = data.get("impacts", [])
        hr_impacts = [i for i in impacts if i.get("domain") == "human_rights"]
        env_impacts = [i for i in impacts if i.get("domain") == "environmental"]
        actual = [i for i in impacts if i.get("impact_type") == "actual"]
        potential = [i for i in impacts if i.get("impact_type") == "potential"]
        risk_scores = []
        for imp in impacts:
            score = _compute_risk_score(
                imp.get("severity", "negligible"),
                imp.get("likelihood", "rare"),
            )
            risk_scores.append(score)
        avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0.0
        return {
            "title": "Impact Summary",
            "total_impacts": len(impacts),
            "human_rights_impacts": len(hr_impacts),
            "environmental_impacts": len(env_impacts),
            "actual_impacts": len(actual),
            "potential_impacts": len(potential),
            "average_risk_score": avg_risk,
            "critical_impacts": sum(1 for s in risk_scores if s >= 20),
            "high_risk_impacts": sum(1 for s in risk_scores if 12 <= s < 20),
            "assessment_date": data.get("assessment_date", utcnow().isoformat()),
        }

    def _section_human_rights_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build human rights impacts section."""
        impacts = data.get("impacts", [])
        hr_impacts = [i for i in impacts if i.get("domain") == "human_rights"]
        entries: List[Dict[str, Any]] = []
        for imp in hr_impacts:
            risk = _compute_risk_score(
                imp.get("severity", "negligible"),
                imp.get("likelihood", "rare"),
            )
            entries.append({
                "impact_id": imp.get("impact_id", ""),
                "category": imp.get("category", ""),
                "description": imp.get("description", ""),
                "impact_type": imp.get("impact_type", "potential"),
                "severity": imp.get("severity", "negligible"),
                "likelihood": imp.get("likelihood", "rare"),
                "risk_score": risk,
                "risk_rating": _risk_rating(risk),
                "affected_groups": imp.get("affected_groups", []),
                "value_chain_location": imp.get("value_chain_location", ""),
                "geographic_scope": imp.get("geographic_scope", []),
                "remediation_status": imp.get("remediation_status", "not_started"),
            })
        entries.sort(key=lambda x: x["risk_score"], reverse=True)
        category_counts: Dict[str, int] = {}
        for e in entries:
            cat = e["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        return {
            "title": "Human Rights Adverse Impacts",
            "total_hr_impacts": len(entries),
            "impacts": entries,
            "category_distribution": category_counts,
        }

    def _section_environmental_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build environmental impacts section."""
        impacts = data.get("impacts", [])
        env_impacts = [i for i in impacts if i.get("domain") == "environmental"]
        entries: List[Dict[str, Any]] = []
        for imp in env_impacts:
            risk = _compute_risk_score(
                imp.get("severity", "negligible"),
                imp.get("likelihood", "rare"),
            )
            entries.append({
                "impact_id": imp.get("impact_id", ""),
                "category": imp.get("category", ""),
                "description": imp.get("description", ""),
                "impact_type": imp.get("impact_type", "potential"),
                "severity": imp.get("severity", "negligible"),
                "likelihood": imp.get("likelihood", "rare"),
                "risk_score": risk,
                "risk_rating": _risk_rating(risk),
                "environmental_medium": imp.get("environmental_medium", ""),
                "value_chain_location": imp.get("value_chain_location", ""),
                "geographic_scope": imp.get("geographic_scope", []),
                "reversibility": imp.get("reversibility", "unknown"),
                "remediation_status": imp.get("remediation_status", "not_started"),
            })
        entries.sort(key=lambda x: x["risk_score"], reverse=True)
        category_counts: Dict[str, int] = {}
        for e in entries:
            cat = e["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        return {
            "title": "Environmental Adverse Impacts",
            "total_env_impacts": len(entries),
            "impacts": entries,
            "category_distribution": category_counts,
        }

    def _section_risk_matrix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build severity x likelihood risk matrix section."""
        impacts = data.get("impacts", [])
        matrix: Dict[str, Dict[str, int]] = {}
        for sev in _SEVERITY_LEVELS:
            matrix[sev] = {}
            for lik in _LIKELIHOOD_LEVELS:
                matrix[sev][lik] = 0
        for imp in impacts:
            sev = imp.get("severity", "negligible")
            lik = imp.get("likelihood", "rare")
            if sev in matrix and lik in matrix.get(sev, {}):
                matrix[sev][lik] += 1
        risk_distribution: Dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0, "negligible": 0,
        }
        for imp in impacts:
            score = _compute_risk_score(
                imp.get("severity", "negligible"),
                imp.get("likelihood", "rare"),
            )
            rating = _risk_rating(score)
            risk_distribution[rating] = risk_distribution.get(rating, 0) + 1
        return {
            "title": "Risk Matrix (Severity x Likelihood)",
            "severity_levels": _SEVERITY_LEVELS,
            "likelihood_levels": _LIKELIHOOD_LEVELS,
            "matrix": matrix,
            "risk_distribution": risk_distribution,
            "total_impacts": len(impacts),
        }

    def _section_prioritisation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build impact prioritisation section per CSDDD Art 7."""
        impacts = data.get("impacts", [])
        scored_impacts: List[Dict[str, Any]] = []
        for imp in impacts:
            risk = _compute_risk_score(
                imp.get("severity", "negligible"),
                imp.get("likelihood", "rare"),
            )
            irremediability = imp.get("irremediability_score", 1)
            scale = imp.get("scale_score", 1)
            priority_score = round(risk * 0.5 + irremediability * 0.3 + scale * 0.2, 2)
            scored_impacts.append({
                "impact_id": imp.get("impact_id", ""),
                "description": imp.get("description", ""),
                "domain": imp.get("domain", ""),
                "risk_score": risk,
                "irremediability_score": irremediability,
                "scale_score": scale,
                "priority_score": priority_score,
                "priority_rank": 0,
            })
        scored_impacts.sort(key=lambda x: x["priority_score"], reverse=True)
        for idx, imp in enumerate(scored_impacts):
            imp["priority_rank"] = idx + 1
        return {
            "title": "Impact Prioritisation (Art 7)",
            "methodology": (
                "Priority score = (Risk x 0.5) + (Irremediability x 0.3) + "
                "(Scale x 0.2). Risk = Severity x Likelihood (1-25)."
            ),
            "total_prioritised": len(scored_impacts),
            "prioritised_impacts": scored_impacts,
        }

    def _section_stakeholder_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build stakeholder impacts section."""
        stakeholders = data.get("stakeholders", [])
        entries: List[Dict[str, Any]] = []
        for sh in stakeholders:
            entries.append({
                "group_name": sh.get("group_name", ""),
                "group_type": sh.get("group_type", ""),
                "estimated_affected": sh.get("estimated_affected", 0),
                "impact_types": sh.get("impact_types", []),
                "vulnerability_level": sh.get("vulnerability_level", "medium"),
                "engagement_status": sh.get("engagement_status", "not_started"),
                "remediation_access": sh.get("remediation_access", False),
            })
        return {
            "title": "Stakeholder Impact Analysis",
            "total_stakeholder_groups": len(entries),
            "stakeholders": entries,
            "total_estimated_affected": sum(
                e.get("estimated_affected", 0) for e in entries
            ),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Adverse Impact Assessment Report - CSDDD\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 6-7"
        )

    def _md_impact_summary(self, data: Dict[str, Any]) -> str:
        """Render impact summary as markdown."""
        sec = self._section_impact_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Impacts Identified | {sec['total_impacts']} |\n"
            f"| Human Rights Impacts | {sec['human_rights_impacts']} |\n"
            f"| Environmental Impacts | {sec['environmental_impacts']} |\n"
            f"| Actual Impacts | {sec['actual_impacts']} |\n"
            f"| Potential Impacts | {sec['potential_impacts']} |\n"
            f"| Critical Risk | {sec['critical_impacts']} |\n"
            f"| High Risk | {sec['high_risk_impacts']} |\n"
            f"| Average Risk Score | {sec['average_risk_score']:.1f} / 25.0 |"
        )

    def _md_human_rights(self, data: Dict[str, Any]) -> str:
        """Render human rights impacts as markdown."""
        sec = self._section_human_rights_impacts(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_hr_impacts']}\n",
            "| Category | Description | Type | Severity | Likelihood | Risk |",
            "|----------|-------------|------|----------|------------|-----:|",
        ]
        for imp in sec["impacts"]:
            lines.append(
                f"| {imp['category']} | {imp['description'][:50]} | "
                f"{imp['impact_type']} | {imp['severity']} | "
                f"{imp['likelihood']} | {imp['risk_score']:.0f} |"
            )
        if sec["category_distribution"]:
            lines.append("\n**Category Distribution:**")
            for cat, count in sec["category_distribution"].items():
                lines.append(f"- {cat.replace('_', ' ').title()}: {count}")
        return "\n".join(lines)

    def _md_environmental(self, data: Dict[str, Any]) -> str:
        """Render environmental impacts as markdown."""
        sec = self._section_environmental_impacts(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_env_impacts']}\n",
            "| Category | Description | Severity | Likelihood | Risk | Reversibility |",
            "|----------|-------------|----------|------------|-----:|---------------|",
        ]
        for imp in sec["impacts"]:
            lines.append(
                f"| {imp['category']} | {imp['description'][:40]} | "
                f"{imp['severity']} | {imp['likelihood']} | "
                f"{imp['risk_score']:.0f} | {imp['reversibility']} |"
            )
        return "\n".join(lines)

    def _md_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render risk matrix as markdown."""
        sec = self._section_risk_matrix(data)
        lines = [
            f"## {sec['title']}\n",
            "| Severity \\ Likelihood | Rare | Unlikely | Possible | Likely | Almost Certain |",
            "|-----------------------|-----:|--------:|--------:|------:|--------------:|",
        ]
        for sev in reversed(_SEVERITY_LEVELS):
            row_vals = [str(sec["matrix"][sev][lik]) for lik in _LIKELIHOOD_LEVELS]
            lines.append(f"| {sev.title()} | {' | '.join(row_vals)} |")
        lines.append(f"\n**Risk Distribution:**")
        for rating, count in sec["risk_distribution"].items():
            lines.append(f"- {rating.title()}: {count}")
        return "\n".join(lines)

    def _md_prioritisation(self, data: Dict[str, Any]) -> str:
        """Render prioritisation as markdown."""
        sec = self._section_prioritisation(data)
        lines = [
            f"## {sec['title']}\n",
            f"*{sec['methodology']}*\n",
            "| Rank | Domain | Description | Risk | Irremediability | Scale | Priority |",
            "|-----:|--------|-------------|-----:|---------------:|------:|---------:|",
        ]
        for imp in sec["prioritised_impacts"][:20]:
            lines.append(
                f"| {imp['priority_rank']} | {imp['domain']} | "
                f"{imp['description'][:40]} | {imp['risk_score']:.0f} | "
                f"{imp['irremediability_score']} | {imp['scale_score']} | "
                f"{imp['priority_score']:.2f} |"
            )
        return "\n".join(lines)

    def _md_stakeholder_impacts(self, data: Dict[str, Any]) -> str:
        """Render stakeholder impacts as markdown."""
        sec = self._section_stakeholder_impacts(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Groups Assessed:** {sec['total_stakeholder_groups']}  \n"
            f"**Total Estimated Affected:** {sec['total_estimated_affected']:,}\n",
            "| Group | Type | Affected | Vulnerability | Engagement |",
            "|-------|------|--------:|---------------|------------|",
        ]
        for sh in sec["stakeholders"]:
            lines.append(
                f"| {sh['group_name']} | {sh['group_type']} | "
                f"{sh['estimated_affected']:,} | {sh['vulnerability_level']} | "
                f"{sh['engagement_status']} |"
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
            ".report{max-width:1200px;margin:auto}"
            "h1{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:.3em}"
            "h2{color:#283593;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".critical{background:#ffcdd2;color:#b71c1c;font-weight:bold}"
            ".high{background:#ffe0b2;color:#e65100}"
            ".medium{background:#fff9c4;color:#f57f17}"
            ".low{background:#c8e6c9;color:#1b5e20}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Adverse Impact Assessment - CSDDD</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_impact_summary(self, data: Dict[str, Any]) -> str:
        """Render impact summary HTML."""
        sec = self._section_impact_summary(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Total Impacts</td><td>{sec['total_impacts']}</td></tr>"
            f"<tr><td>Human Rights</td><td>{sec['human_rights_impacts']}</td></tr>"
            f"<tr><td>Environmental</td><td>{sec['environmental_impacts']}</td></tr>"
            f"<tr><td>Critical Risk</td><td class='critical'>{sec['critical_impacts']}</td></tr>"
            f"<tr><td>Avg Risk Score</td><td>{sec['average_risk_score']:.1f}</td></tr>"
            f"</table>"
        )

    def _html_human_rights(self, data: Dict[str, Any]) -> str:
        """Render human rights impacts HTML."""
        sec = self._section_human_rights_impacts(data)
        rows = ""
        for imp in sec["impacts"][:15]:
            css_class = imp["risk_rating"]
            rows += (
                f"<tr class='{css_class}'><td>{imp['category']}</td>"
                f"<td>{imp['description'][:60]}</td>"
                f"<td>{imp['severity']}</td>"
                f"<td>{imp['risk_score']:.0f}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_hr_impacts']}</p>\n"
            f"<table><tr><th>Category</th><th>Description</th>"
            f"<th>Severity</th><th>Risk</th></tr>{rows}</table>"
        )

    def _html_environmental(self, data: Dict[str, Any]) -> str:
        """Render environmental impacts HTML."""
        sec = self._section_environmental_impacts(data)
        rows = ""
        for imp in sec["impacts"][:15]:
            css_class = imp["risk_rating"]
            rows += (
                f"<tr class='{css_class}'><td>{imp['category']}</td>"
                f"<td>{imp['description'][:60]}</td>"
                f"<td>{imp['severity']}</td>"
                f"<td>{imp['risk_score']:.0f}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_env_impacts']}</p>\n"
            f"<table><tr><th>Category</th><th>Description</th>"
            f"<th>Severity</th><th>Risk</th></tr>{rows}</table>"
        )

    def _html_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render risk matrix HTML."""
        sec = self._section_risk_matrix(data)
        header = "<tr><th>Severity \\ Likelihood</th>"
        for lik in _LIKELIHOOD_LEVELS:
            header += f"<th>{lik.replace('_', ' ').title()}</th>"
        header += "</tr>"
        rows = ""
        for sev in reversed(_SEVERITY_LEVELS):
            rows += f"<tr><td><strong>{sev.title()}</strong></td>"
            for lik in _LIKELIHOOD_LEVELS:
                count = sec["matrix"][sev][lik]
                score = _compute_risk_score(sev, lik)
                css_class = _risk_rating(score)
                rows += f"<td class='{css_class}'>{count}</td>"
            rows += "</tr>"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table>{header}{rows}</table>"
        )

    def _html_prioritisation(self, data: Dict[str, Any]) -> str:
        """Render prioritisation HTML."""
        sec = self._section_prioritisation(data)
        rows = "".join(
            f"<tr><td>{imp['priority_rank']}</td><td>{imp['domain']}</td>"
            f"<td>{imp['description'][:60]}</td>"
            f"<td>{imp['priority_score']:.2f}</td></tr>"
            for imp in sec["prioritised_impacts"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><em>{sec['methodology']}</em></p>\n"
            f"<table><tr><th>Rank</th><th>Domain</th><th>Description</th>"
            f"<th>Priority</th></tr>{rows}</table>"
        )
