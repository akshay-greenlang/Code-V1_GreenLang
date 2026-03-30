# -*- coding: utf-8 -*-
"""
S3CommunitiesReportTemplate - ESRS S3 Affected Communities Report

Renders community policies, engagement processes, FPIC processes,
remediation channels, actions and impacts, community assessments,
targets, and grievance metrics per ESRS S3.

Sections:
    1. Policies
    2. Engagement Processes
    3. FPIC Processes
    4. Remediation Channels
    5. Actions and Impacts
    6. Community Assessments
    7. Targets
    8. Grievance Metrics

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
    "policies", "engagement_processes", "fpic_processes", "remediation_channels",
    "actions_and_impacts", "community_assessments", "targets", "grievance_metrics",
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

class S3CommunitiesReportTemplate:
    """
    ESRS S3 Affected Communities report template.

    Renders community engagement policies, FPIC processes, remediation
    channels, impact assessments, community-level assessments, targets,
    and grievance metrics per ESRS S3.

    Example:
        >>> tpl = S3CommunitiesReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize S3CommunitiesReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

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
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "community_assessments" not in data:
            warnings.append("community_assessments missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render S3 Affected Communities report as Markdown."""
        self.generated_at = utcnow()
        sections = [self._md_header(data), self._md_policies(data), self._md_engagement(data),
                     self._md_fpic(data), self._md_remediation(data), self._md_actions_impacts(data),
                     self._md_assessments(data), self._md_targets(data), self._md_grievance(data),
                     self._md_footer(data)]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render S3 Affected Communities report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([self._html_header(data), self._html_assessments(data), self._html_grievance(data)])
        prov = _compute_hash(body)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ESRS S3 Affected Communities Report</title>\n<style>\n{css}\n</style>\n'
                f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
                f'<!-- Provenance: {prov} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render S3 Affected Communities report as JSON string."""
        self.generated_at = utcnow()
        result = {"template": "s3_communities_report", "esrs_reference": "ESRS S3", "version": "17.0.0",
                  "generated_at": self.generated_at.isoformat(), "entity_name": data.get("entity_name", ""),
                  "reporting_year": data.get("reporting_year", ""),
                  "communities_assessed": len(data.get("community_assessments", [])),
                  "fpic_applied": data.get("fpic_applied", False),
                  "grievances_filed": data.get("community_grievances_filed", 0)}
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # -- Section renderers --

    def _section_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        policies = data.get("community_policies", [])
        return {"title": "Community Policies", "policy_count": len(policies),
                "policies": [{"name": p.get("name", ""), "scope": p.get("scope", ""),
                              "indigenous_rights": p.get("indigenous_rights", False),
                              "land_rights": p.get("land_rights", False)} for p in policies]}

    def _section_engagement_processes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Community Engagement Processes",
                "engagement_methods": data.get("community_engagement_methods", []),
                "indigenous_engagement": data.get("indigenous_engagement", False),
                "frequency": data.get("community_engagement_frequency", "")}

    def _section_fpic_processes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Free, Prior and Informed Consent (FPIC)", "fpic_applied": data.get("fpic_applied", False),
                "fpic_description": data.get("fpic_description", ""),
                "projects_requiring_fpic": data.get("projects_requiring_fpic", 0),
                "fpic_obtained_count": data.get("fpic_obtained_count", 0)}

    def _section_remediation_channels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        channels = data.get("community_remediation_channels", [])
        return {"title": "Remediation Channels", "channel_count": len(channels),
                "channels": [{"name": c.get("name", ""), "type": c.get("type", ""),
                              "community_accessible": c.get("community_accessible", False)} for c in channels]}

    def _section_actions_and_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        actions = data.get("community_actions", [])
        impacts = data.get("community_impacts", [])
        return {"title": "Actions and Impacts", "action_count": len(actions), "impact_count": len(impacts),
                "actions": [{"description": a.get("description", ""), "status": a.get("status", ""),
                             "investment_eur": a.get("investment_eur", 0.0)} for a in actions],
                "impacts": [{"description": i.get("description", ""), "type": i.get("type", ""),
                             "severity": i.get("severity", ""),
                             "affected_population": i.get("affected_population", 0)} for i in impacts]}

    def _section_community_assessments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        assessments = data.get("community_assessments", [])
        return {"title": "Community Assessments", "total_assessed": len(assessments),
                "assessments": [{"community_name": a.get("community_name", ""), "location": a.get("location", ""),
                                 "population": a.get("population", 0), "impact_type": a.get("impact_type", ""),
                                 "risk_level": a.get("risk_level", ""),
                                 "mitigation": a.get("mitigation", "")} for a in assessments]}

    def _section_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        targets = data.get("community_targets", [])
        return {"title": "Community Targets", "target_count": len(targets),
                "targets": [{"name": t.get("name", ""), "metric": t.get("metric", ""),
                             "target_year": t.get("target_year", ""), "target_value": t.get("target_value", 0.0),
                             "current_value": t.get("current_value", 0.0)} for t in targets]}

    def _section_grievance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Grievance Metrics", "grievances_filed": data.get("community_grievances_filed", 0),
                "grievances_resolved": data.get("community_grievances_resolved", 0),
                "avg_resolution_days": data.get("community_avg_resolution_days", 0),
                "community_compensation_eur": data.get("community_compensation_eur", 0.0)}

    # -- Markdown helpers --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# ESRS S3 Affected Communities Report\n\n**Entity:** {data.get('entity_name', '')}  \n"
                f"**Reporting Year:** {data.get('reporting_year', '')}  \n**Generated:** {ts}  \n"
                f"**Standard:** ESRS S3 Affected Communities")

    def _md_policies(self, d: Dict[str, Any]) -> str:
        sec = self._section_policies(d)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            ir = "Yes" if p["indigenous_rights"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, Indigenous Rights: {ir})")
        return "\n".join(lines)

    def _md_engagement(self, d: Dict[str, Any]) -> str:
        sec = self._section_engagement_processes(d)
        ind = "Yes" if sec["indigenous_engagement"] else "No"
        return f"## {sec['title']}\n\n**Indigenous Engagement:** {ind}  \n**Frequency:** {sec['frequency']}"

    def _md_fpic(self, d: Dict[str, Any]) -> str:
        sec = self._section_fpic_processes(d)
        applied = "Yes" if sec["fpic_applied"] else "No"
        return (f"## {sec['title']}\n\n**FPIC Applied:** {applied}  \n"
                f"**Projects Requiring FPIC:** {sec['projects_requiring_fpic']}  \n"
                f"**FPIC Obtained:** {sec['fpic_obtained_count']}\n\n{sec['fpic_description']}")

    def _md_remediation(self, d: Dict[str, Any]) -> str:
        sec = self._section_remediation_channels(d)
        lines = [f"## {sec['title']}\n"]
        for c in sec["channels"]:
            acc = "Yes" if c["community_accessible"] else "No"
            lines.append(f"- **{c['name']}** ({c['type']}) - Accessible: {acc}")
        return "\n".join(lines)

    def _md_actions_impacts(self, d: Dict[str, Any]) -> str:
        sec = self._section_actions_and_impacts(d)
        lines = [f"## {sec['title']}\n"]
        if sec["impacts"]:
            lines.extend(["| Impact | Type | Severity | Affected Population |",
                          "|--------|------|----------|-------------------:|"])
            for i in sec["impacts"]:
                lines.append(f"| {i['description']} | {i['type']} | {i['severity']} | {i['affected_population']:,} |")
        return "\n".join(lines)

    def _md_assessments(self, d: Dict[str, Any]) -> str:
        sec = self._section_community_assessments(d)
        lines = [f"## {sec['title']}\n", f"**Communities Assessed:** {sec['total_assessed']}\n"]
        if sec["assessments"]:
            lines.extend(["| Community | Location | Population | Risk | Impact |",
                          "|-----------|----------|----------:|------|--------|"])
            for a in sec["assessments"]:
                lines.append(f"| {a['community_name']} | {a['location']} | {a['population']:,} | {a['risk_level']} | {a['impact_type']} |")
        return "\n".join(lines)

    def _md_targets(self, d: Dict[str, Any]) -> str:
        sec = self._section_targets(d)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.extend(["| Target | Metric | Current | Goal | Year |", "|--------|--------|--------:|-----:|-----:|"])
            for t in sec["targets"]:
                lines.append(f"| {t['name']} | {t['metric']} | {t['current_value']:.1f} | {t['target_value']:.1f} | {t['target_year']} |")
        return "\n".join(lines)

    def _md_grievance(self, d: Dict[str, Any]) -> str:
        sec = self._section_grievance_metrics(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Filed | {sec['grievances_filed']} |\n| Resolved | {sec['grievances_resolved']} |\n"
                f"| Avg Resolution (days) | {sec['avg_resolution_days']} |\n"
                f"| Compensation (EUR) | {sec['community_compensation_eur']:,.2f} |")

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    def _css(self) -> str:
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}.report{max-width:900px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.total{font-weight:bold;background:#e8f5e9}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        return (f"<h1>ESRS S3 Affected Communities Report</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_assessments(self, data: Dict[str, Any]) -> str:
        sec = self._section_community_assessments(data)
        rows = "".join(f"<tr><td>{a['community_name']}</td><td>{a['location']}</td><td>{a['risk_level']}</td></tr>"
                       for a in sec["assessments"])
        return (f"<h2>{sec['title']}</h2>\n"
                f"<table><tr><th>Community</th><th>Location</th><th>Risk</th></tr>{rows}</table>")

    def _html_grievance(self, data: Dict[str, Any]) -> str:
        sec = self._section_grievance_metrics(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Filed</td><td>{sec['grievances_filed']}</td></tr>"
                f"<tr><td>Resolved</td><td>{sec['grievances_resolved']}</td></tr></table>")

# Alias for backward compatibility with templates/__init__.py
S3CommunitiesReport = S3CommunitiesReportTemplate
