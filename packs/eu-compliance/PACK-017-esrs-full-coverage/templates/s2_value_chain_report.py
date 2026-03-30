# -*- coding: utf-8 -*-
"""
S2ValueChainReportTemplate - ESRS S2 Workers in the Value Chain Report

Renders value chain worker policies, engagement processes, remediation
channels, actions and risks, due diligence overview, supplier risk
assessment, targets, and grievance metrics per ESRS S2.

Sections:
    1. Policies
    2. Engagement Processes
    3. Remediation Channels
    4. Actions and Risks
    5. Due Diligence Overview
    6. Supplier Risk Assessment
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
    "policies", "engagement_processes", "remediation_channels",
    "actions_and_risks", "due_diligence_overview",
    "supplier_risk_assessment", "targets", "grievance_metrics",
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

class S2ValueChainReportTemplate:
    """
    ESRS S2 Workers in the Value Chain report template.

    Renders policies, engagement, remediation, due diligence processes,
    supplier risk assessments, measurable targets, and grievance metrics
    for workers in the undertaking's value chain per ESRS S2.

    Example:
        >>> tpl = S2ValueChainReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize S2ValueChainReportTemplate."""
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
        if "supplier_assessments" not in data:
            warnings.append("supplier_assessments missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render S2 Value Chain Workers report as Markdown."""
        self.generated_at = utcnow()
        sections = [self._md_header(data), self._md_policies(data), self._md_engagement(data),
                     self._md_remediation(data), self._md_actions_risks(data),
                     self._md_due_diligence(data), self._md_supplier_risk(data),
                     self._md_targets(data), self._md_grievance(data), self._md_footer(data)]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render S2 Value Chain Workers report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([self._html_header(data), self._html_supplier_risk(data), self._html_grievance(data)])
        prov = _compute_hash(body)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ESRS S2 Value Chain Workers Report</title>\n<style>\n{css}\n</style>\n'
                f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
                f'<!-- Provenance: {prov} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render S2 Value Chain Workers report as JSON string."""
        self.generated_at = utcnow()
        result = {
            "template": "s2_value_chain_report", "esrs_reference": "ESRS S2",
            "version": "17.0.0", "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""), "reporting_year": data.get("reporting_year", ""),
            "suppliers_assessed": len(data.get("supplier_assessments", [])),
            "high_risk_suppliers": data.get("high_risk_supplier_count", 0),
            "grievances_filed": data.get("grievances_filed", 0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # -- Section renderers --

    def _section_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build policies section."""
        policies = data.get("vc_worker_policies", [])
        return {"title": "Value Chain Worker Policies", "policy_count": len(policies),
                "policies": [{"name": p.get("name", ""), "scope": p.get("scope", ""),
                              "human_rights_aligned": p.get("human_rights_aligned", False),
                              "ilo_conventions": p.get("ilo_conventions", [])} for p in policies]}

    def _section_engagement_processes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement processes section."""
        return {"title": "Engagement Processes", "engagement_methods": data.get("vc_engagement_methods", []),
                "worker_representatives_consulted": data.get("worker_reps_consulted", False),
                "frequency": data.get("vc_engagement_frequency", "")}

    def _section_remediation_channels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remediation channels section."""
        channels = data.get("vc_remediation_channels", [])
        return {"title": "Remediation Channels", "channel_count": len(channels),
                "channels": [{"name": c.get("name", ""), "type": c.get("type", ""),
                              "accessible_to_vc_workers": c.get("accessible_to_vc_workers", False)} for c in channels]}

    def _section_actions_and_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build actions and risks section."""
        actions = data.get("vc_actions", [])
        risks = data.get("vc_worker_risks", [])
        return {"title": "Actions and Risks", "action_count": len(actions), "risk_count": len(risks),
                "actions": [{"description": a.get("description", ""), "status": a.get("status", "")} for a in actions],
                "risks": [{"risk": r.get("risk", ""), "severity": r.get("severity", ""),
                           "affected_group": r.get("affected_group", "")} for r in risks]}

    def _section_due_diligence_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build due diligence overview section."""
        return {"title": "Due Diligence Overview", "dd_process_description": data.get("vc_dd_process", ""),
                "ung_principles_aligned": data.get("ungp_aligned", False),
                "oecd_guidelines_aligned": data.get("oecd_aligned", False),
                "tiers_covered": data.get("supply_chain_tiers_covered", 0)}

    def _section_supplier_risk_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build supplier risk assessment section."""
        assessments = data.get("supplier_assessments", [])
        return {"title": "Supplier Risk Assessment", "total_assessed": len(assessments),
                "high_risk_count": sum(1 for a in assessments if a.get("risk_level") == "high"),
                "medium_risk_count": sum(1 for a in assessments if a.get("risk_level") == "medium"),
                "low_risk_count": sum(1 for a in assessments if a.get("risk_level") == "low"),
                "assessments": [{"supplier": a.get("supplier", ""), "country": a.get("country", ""),
                                 "risk_level": a.get("risk_level", ""), "issues": a.get("issues", [])}
                                for a in assessments[:20]]}

    def _section_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build targets section."""
        targets = data.get("vc_targets", [])
        return {"title": "Value Chain Worker Targets", "target_count": len(targets),
                "targets": [{"name": t.get("name", ""), "metric": t.get("metric", ""),
                             "target_year": t.get("target_year", ""), "target_value": t.get("target_value", 0.0),
                             "current_value": t.get("current_value", 0.0)} for t in targets]}

    def _section_grievance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build grievance metrics section."""
        return {"title": "Grievance Metrics", "grievances_filed": data.get("grievances_filed", 0),
                "grievances_resolved": data.get("grievances_resolved", 0),
                "avg_resolution_days": data.get("avg_resolution_days", 0),
                "remediation_provided": data.get("remediation_provided", 0)}

    # -- Markdown helpers --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# ESRS S2 Workers in the Value Chain Report\n\n**Entity:** {data.get('entity_name', '')}  \n"
                f"**Reporting Year:** {data.get('reporting_year', '')}  \n**Generated:** {ts}  \n"
                f"**Standard:** ESRS S2 Workers in the Value Chain")

    def _md_policies(self, d: Dict[str, Any]) -> str:
        sec = self._section_policies(d)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            hr = "Yes" if p["human_rights_aligned"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, HR Aligned: {hr})")
        return "\n".join(lines)

    def _md_engagement(self, d: Dict[str, Any]) -> str:
        sec = self._section_engagement_processes(d)
        reps = "Yes" if sec["worker_representatives_consulted"] else "No"
        return f"## {sec['title']}\n\n**Worker Reps Consulted:** {reps}  \n**Frequency:** {sec['frequency']}"

    def _md_remediation(self, d: Dict[str, Any]) -> str:
        sec = self._section_remediation_channels(d)
        lines = [f"## {sec['title']}\n"]
        for c in sec["channels"]:
            acc = "Yes" if c["accessible_to_vc_workers"] else "No"
            lines.append(f"- **{c['name']}** ({c['type']}) - VC Accessible: {acc}")
        return "\n".join(lines)

    def _md_actions_risks(self, d: Dict[str, Any]) -> str:
        sec = self._section_actions_and_risks(d)
        lines = [f"## {sec['title']}\n", f"**Actions:** {sec['action_count']} | **Risks:** {sec['risk_count']}\n"]
        if sec["risks"]:
            lines.extend(["| Risk | Severity | Affected Group |", "|------|----------|----------------|"])
            for r in sec["risks"]:
                lines.append(f"| {r['risk']} | {r['severity']} | {r['affected_group']} |")
        return "\n".join(lines)

    def _md_due_diligence(self, d: Dict[str, Any]) -> str:
        sec = self._section_due_diligence_overview(d)
        ungp = "Yes" if sec["ung_principles_aligned"] else "No"
        oecd = "Yes" if sec["oecd_guidelines_aligned"] else "No"
        return (f"## {sec['title']}\n\n**UNGP Aligned:** {ungp}  \n**OECD Aligned:** {oecd}  \n"
                f"**Supply Chain Tiers Covered:** {sec['tiers_covered']}")

    def _md_supplier_risk(self, d: Dict[str, Any]) -> str:
        sec = self._section_supplier_risk_assessment(d)
        lines = [f"## {sec['title']}\n",
                 f"**Assessed:** {sec['total_assessed']} | **High:** {sec['high_risk_count']} "
                 f"| **Medium:** {sec['medium_risk_count']} | **Low:** {sec['low_risk_count']}\n"]
        if sec["assessments"]:
            lines.extend(["| Supplier | Country | Risk Level |", "|----------|---------|------------|"])
            for a in sec["assessments"]:
                lines.append(f"| {a['supplier']} | {a['country']} | {a['risk_level']} |")
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
                f"| Remediation Provided | {sec['remediation_provided']} |")

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    def _css(self) -> str:
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}.report{max-width:900px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.total{font-weight:bold;background:#e8f5e9}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        return (f"<h1>ESRS S2 Value Chain Workers Report</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_supplier_risk(self, data: Dict[str, Any]) -> str:
        sec = self._section_supplier_risk_assessment(data)
        rows = "".join(f"<tr><td>{a['supplier']}</td><td>{a['country']}</td><td>{a['risk_level']}</td></tr>"
                       for a in sec["assessments"])
        return (f"<h2>{sec['title']}</h2>\n<p>Assessed: {sec['total_assessed']} | High Risk: {sec['high_risk_count']}</p>\n"
                f"<table><tr><th>Supplier</th><th>Country</th><th>Risk</th></tr>{rows}</table>")

    def _html_grievance(self, data: Dict[str, Any]) -> str:
        sec = self._section_grievance_metrics(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Filed</td><td>{sec['grievances_filed']}</td></tr>"
                f"<tr><td>Resolved</td><td>{sec['grievances_resolved']}</td></tr></table>")

# Alias for backward compatibility with templates/__init__.py
S2ValueChainReport = S2ValueChainReportTemplate
