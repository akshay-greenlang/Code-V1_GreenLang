# -*- coding: utf-8 -*-
"""
S4ConsumersReportTemplate - ESRS S4 Consumers and End-Users Report

Renders consumer policies, engagement channels, remediation processes,
actions overview, product safety, data privacy, vulnerable consumers,
targets, and complaint metrics per ESRS S4.

Sections:
    1. Policies
    2. Engagement Channels
    3. Remediation Processes
    4. Actions Overview
    5. Product Safety
    6. Data Privacy
    7. Vulnerable Consumers
    8. Targets
    9. Complaint Metrics

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "policies", "engagement_channels", "remediation_processes",
    "actions_overview", "product_safety", "data_privacy",
    "vulnerable_consumers", "targets", "complaint_metrics",
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


class S4ConsumersReportTemplate:
    """
    ESRS S4 Consumers and End-Users report template.

    Renders consumer protection policies, engagement channels, remediation
    processes, product safety metrics, data privacy compliance, vulnerable
    consumer protections, targets, and complaint metrics per ESRS S4.

    Example:
        >>> tpl = S4ConsumersReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize S4ConsumersReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

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
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "product_safety_incidents" not in data:
            warnings.append("product_safety_incidents missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render S4 Consumers report as Markdown."""
        self.generated_at = _utcnow()
        sections = [self._md_header(data), self._md_policies(data), self._md_engagement(data),
                     self._md_remediation(data), self._md_actions(data), self._md_product_safety(data),
                     self._md_data_privacy(data), self._md_vulnerable(data), self._md_targets(data),
                     self._md_complaints(data), self._md_footer(data)]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render S4 Consumers report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([self._html_header(data), self._html_product_safety(data),
                          self._html_data_privacy(data), self._html_complaints(data)])
        prov = _compute_hash(body)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ESRS S4 Consumers Report</title>\n<style>\n{css}\n</style>\n'
                f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
                f'<!-- Provenance: {prov} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render S4 Consumers report as JSON string."""
        self.generated_at = _utcnow()
        result = {"template": "s4_consumers_report", "esrs_reference": "ESRS S4", "version": "17.0.0",
                  "generated_at": self.generated_at.isoformat(), "entity_name": data.get("entity_name", ""),
                  "reporting_year": data.get("reporting_year", ""),
                  "product_recalls": data.get("product_recalls", 0),
                  "data_breaches": data.get("data_breaches", 0),
                  "complaints_received": data.get("complaints_received", 0)}
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # -- Section renderers --

    def _section_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        policies = data.get("consumer_policies", [])
        return {"title": "Consumer and End-User Policies", "policy_count": len(policies),
                "policies": [{"name": p.get("name", ""), "scope": p.get("scope", ""),
                              "consumer_rights_aligned": p.get("consumer_rights_aligned", False)} for p in policies]}

    def _section_engagement_channels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        channels = data.get("consumer_engagement_channels", [])
        return {"title": "Consumer Engagement Channels", "channel_count": len(channels),
                "channels": [{"name": c.get("name", ""), "type": c.get("type", ""),
                              "digital": c.get("digital", False), "languages": c.get("languages", [])} for c in channels]}

    def _section_remediation_processes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Remediation Processes", "has_recall_procedure": data.get("has_recall_procedure", False),
                "has_complaint_mechanism": data.get("has_complaint_mechanism", False),
                "avg_resolution_days": data.get("consumer_avg_resolution_days", 0),
                "compensation_policy": data.get("compensation_policy", "")}

    def _section_actions_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        actions = data.get("consumer_actions", [])
        return {"title": "Consumer Protection Actions", "action_count": len(actions),
                "actions": [{"description": a.get("description", ""), "status": a.get("status", ""),
                             "investment_eur": a.get("investment_eur", 0.0)} for a in actions]}

    def _section_product_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        incidents = data.get("product_safety_incidents", [])
        return {"title": "Product Safety", "incident_count": len(incidents),
                "product_recalls": data.get("product_recalls", 0),
                "safety_certifications": data.get("safety_certifications", []),
                "incidents": [{"product": i.get("product", ""), "type": i.get("type", ""),
                               "severity": i.get("severity", ""), "corrective_action": i.get("corrective_action", "")}
                              for i in incidents]}

    def _section_data_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Data Privacy and Security", "gdpr_compliant": data.get("gdpr_compliant", False),
                "data_breaches": data.get("data_breaches", 0),
                "affected_individuals": data.get("breach_affected_individuals", 0),
                "dpo_appointed": data.get("dpo_appointed", False),
                "privacy_impact_assessments": data.get("privacy_impact_assessments", 0),
                "data_subject_requests": data.get("data_subject_requests", 0)}

    def _section_vulnerable_consumers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Vulnerable Consumer Protections",
                "has_vulnerability_policy": data.get("has_vulnerability_policy", False),
                "accessibility_features": data.get("accessibility_features", []),
                "age_verification": data.get("age_verification", False),
                "plain_language_communications": data.get("plain_language_communications", False)}

    def _section_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        targets = data.get("consumer_targets", [])
        return {"title": "Consumer Protection Targets", "target_count": len(targets),
                "targets": [{"name": t.get("name", ""), "metric": t.get("metric", ""),
                             "target_year": t.get("target_year", ""), "target_value": t.get("target_value", 0.0),
                             "current_value": t.get("current_value", 0.0)} for t in targets]}

    def _section_complaint_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Complaint Metrics", "complaints_received": data.get("complaints_received", 0),
                "complaints_resolved": data.get("complaints_resolved", 0),
                "avg_resolution_days": data.get("complaint_avg_resolution_days", 0),
                "satisfaction_rate_pct": round(data.get("complaint_satisfaction_pct", 0.0), 1),
                "regulatory_complaints": data.get("regulatory_complaints", 0)}

    # -- Markdown helpers --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# ESRS S4 Consumers and End-Users Report\n\n**Entity:** {data.get('entity_name', '')}  \n"
                f"**Reporting Year:** {data.get('reporting_year', '')}  \n**Generated:** {ts}  \n"
                f"**Standard:** ESRS S4 Consumers and End-Users")

    def _md_policies(self, d: Dict[str, Any]) -> str:
        sec = self._section_policies(d)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            cr = "Yes" if p["consumer_rights_aligned"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, Consumer Rights: {cr})")
        return "\n".join(lines)

    def _md_engagement(self, d: Dict[str, Any]) -> str:
        sec = self._section_engagement_channels(d)
        lines = [f"## {sec['title']}\n"]
        for c in sec["channels"]:
            dig = "Digital" if c["digital"] else "Physical"
            lines.append(f"- **{c['name']}** ({c['type']}, {dig})")
        return "\n".join(lines)

    def _md_remediation(self, d: Dict[str, Any]) -> str:
        sec = self._section_remediation_processes(d)
        recall = "Yes" if sec["has_recall_procedure"] else "No"
        complaint = "Yes" if sec["has_complaint_mechanism"] else "No"
        return (f"## {sec['title']}\n\n**Recall Procedure:** {recall}  \n**Complaint Mechanism:** {complaint}  \n"
                f"**Avg Resolution:** {sec['avg_resolution_days']} days")

    def _md_actions(self, d: Dict[str, Any]) -> str:
        sec = self._section_actions_overview(d)
        lines = [f"## {sec['title']}\n"]
        if sec["actions"]:
            lines.extend(["| Action | Status | Investment (EUR) |", "|--------|--------|----------------:|"])
            for a in sec["actions"]:
                lines.append(f"| {a['description']} | {a['status']} | {a['investment_eur']:,.2f} |")
        return "\n".join(lines)

    def _md_product_safety(self, d: Dict[str, Any]) -> str:
        sec = self._section_product_safety(d)
        lines = [f"## {sec['title']}\n", f"**Incidents:** {sec['incident_count']} | **Recalls:** {sec['product_recalls']}\n"]
        if sec["incidents"]:
            lines.extend(["| Product | Type | Severity | Corrective Action |",
                          "|---------|------|----------|-------------------|"])
            for i in sec["incidents"]:
                lines.append(f"| {i['product']} | {i['type']} | {i['severity']} | {i['corrective_action']} |")
        return "\n".join(lines)

    def _md_data_privacy(self, d: Dict[str, Any]) -> str:
        sec = self._section_data_privacy(d)
        gdpr = "Yes" if sec["gdpr_compliant"] else "No"
        dpo = "Yes" if sec["dpo_appointed"] else "No"
        return (f"## {sec['title']}\n\n**GDPR Compliant:** {gdpr}  \n**DPO Appointed:** {dpo}  \n"
                f"**Data Breaches:** {sec['data_breaches']}  \n**Affected Individuals:** {sec['affected_individuals']:,}  \n"
                f"**PIAs Conducted:** {sec['privacy_impact_assessments']}  \n"
                f"**Data Subject Requests:** {sec['data_subject_requests']}")

    def _md_vulnerable(self, d: Dict[str, Any]) -> str:
        sec = self._section_vulnerable_consumers(d)
        vuln = "Yes" if sec["has_vulnerability_policy"] else "No"
        age = "Yes" if sec["age_verification"] else "No"
        plain = "Yes" if sec["plain_language_communications"] else "No"
        return (f"## {sec['title']}\n\n**Vulnerability Policy:** {vuln}  \n"
                f"**Age Verification:** {age}  \n**Plain Language:** {plain}")

    def _md_targets(self, d: Dict[str, Any]) -> str:
        sec = self._section_targets(d)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.extend(["| Target | Metric | Current | Goal | Year |", "|--------|--------|--------:|-----:|-----:|"])
            for t in sec["targets"]:
                lines.append(f"| {t['name']} | {t['metric']} | {t['current_value']:.1f} | {t['target_value']:.1f} | {t['target_year']} |")
        return "\n".join(lines)

    def _md_complaints(self, d: Dict[str, Any]) -> str:
        sec = self._section_complaint_metrics(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Received | {sec['complaints_received']} |\n| Resolved | {sec['complaints_resolved']} |\n"
                f"| Avg Resolution (days) | {sec['avg_resolution_days']} |\n"
                f"| Satisfaction Rate | {sec['satisfaction_rate_pct']:.1f}% |\n"
                f"| Regulatory Complaints | {sec['regulatory_complaints']} |")

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    def _css(self) -> str:
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}.report{max-width:900px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.total{font-weight:bold;background:#e8f5e9}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        return (f"<h1>ESRS S4 Consumers Report</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_product_safety(self, data: Dict[str, Any]) -> str:
        sec = self._section_product_safety(data)
        rows = "".join(f"<tr><td>{i['product']}</td><td>{i['type']}</td><td>{i['severity']}</td></tr>"
                       for i in sec["incidents"])
        return (f"<h2>{sec['title']}</h2>\n<p>Incidents: {sec['incident_count']} | Recalls: {sec['product_recalls']}</p>\n"
                f"<table><tr><th>Product</th><th>Type</th><th>Severity</th></tr>{rows}</table>")

    def _html_data_privacy(self, data: Dict[str, Any]) -> str:
        sec = self._section_data_privacy(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>GDPR</td><td>{'Yes' if sec['gdpr_compliant'] else 'No'}</td></tr>"
                f"<tr><td>Breaches</td><td>{sec['data_breaches']}</td></tr></table>")

    def _html_complaints(self, data: Dict[str, Any]) -> str:
        sec = self._section_complaint_metrics(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Received</td><td>{sec['complaints_received']}</td></tr>"
                f"<tr><td>Resolved</td><td>{sec['complaints_resolved']}</td></tr></table>")


# Alias for backward compatibility with templates/__init__.py
S4ConsumersReport = S4ConsumersReportTemplate
