# -*- coding: utf-8 -*-
"""
G1GovernanceReportTemplate - ESRS G1 Business Conduct Report

Renders business conduct policies, supplier management, anti-corruption
training, corruption incidents, political influence, lobbying expenditure,
payment practices, SME payment terms, and targets per ESRS G1.

Sections:
    1. Conduct Policies
    2. Supplier Management
    3. Anti-Corruption Training
    4. Corruption Incidents
    5. Political Influence
    6. Lobbying Expenditure
    7. Payment Practices
    8. SME Payment Terms
    9. Targets

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
    "conduct_policies", "supplier_management", "anti_corruption_training",
    "corruption_incidents", "political_influence", "lobbying_expenditure",
    "payment_practices", "sme_payment_terms", "targets",
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


class G1GovernanceReportTemplate:
    """
    ESRS G1 Business Conduct report template.

    Renders corporate conduct policies, supplier relationship management,
    anti-corruption training, corruption and bribery incidents, political
    engagement, lobbying expenditure, payment practices including SME
    terms, and measurable targets per ESRS G1.

    Example:
        >>> tpl = G1GovernanceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize G1GovernanceReportTemplate."""
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
        if "corruption_incidents" not in data:
            warnings.append("corruption_incidents missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render G1 Business Conduct report as Markdown."""
        self.generated_at = _utcnow()
        sections = [self._md_header(data), self._md_conduct_policies(data), self._md_supplier_mgmt(data),
                     self._md_anti_corruption(data), self._md_corruption_incidents(data),
                     self._md_political(data), self._md_lobbying(data), self._md_payment(data),
                     self._md_sme_payment(data), self._md_targets(data), self._md_footer(data)]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render G1 Business Conduct report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([self._html_header(data), self._html_corruption(data),
                          self._html_payment(data), self._html_lobbying(data)])
        prov = _compute_hash(body)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>ESRS G1 Business Conduct Report</title>\n<style>\n{css}\n</style>\n'
                f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
                f'<!-- Provenance: {prov} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render G1 Business Conduct report as JSON string."""
        self.generated_at = _utcnow()
        result = {"template": "g1_governance_report", "esrs_reference": "ESRS G1", "version": "17.0.0",
                  "generated_at": self.generated_at.isoformat(), "entity_name": data.get("entity_name", ""),
                  "reporting_year": data.get("reporting_year", ""),
                  "corruption_incidents_count": len(data.get("corruption_incidents", [])),
                  "lobbying_total_eur": data.get("lobbying_total_eur", 0.0),
                  "avg_payment_days": data.get("avg_payment_days", 0)}
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # -- Section renderers --

    def _section_conduct_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        policies = data.get("conduct_policies", [])
        return {"title": "Business Conduct Policies", "policy_count": len(policies),
                "has_code_of_conduct": data.get("has_code_of_conduct", False),
                "whistleblower_protection": data.get("whistleblower_protection", False),
                "policies": [{"name": p.get("name", ""), "scope": p.get("scope", ""),
                              "covers_value_chain": p.get("covers_value_chain", False)} for p in policies]}

    def _section_supplier_management(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Supplier Relationship Management",
                "supplier_code_of_conduct": data.get("supplier_code_of_conduct", False),
                "suppliers_screened_pct": round(data.get("suppliers_screened_pct", 0.0), 1),
                "suppliers_audited": data.get("suppliers_audited", 0),
                "terminated_for_violations": data.get("suppliers_terminated", 0)}

    def _section_anti_corruption_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Anti-Corruption Training",
                "employees_trained_pct": round(data.get("anti_corruption_trained_pct", 0.0), 1),
                "board_trained_pct": round(data.get("board_anti_corruption_trained_pct", 0.0), 1),
                "training_hours_total": data.get("anti_corruption_training_hours", 0),
                "training_frequency": data.get("anti_corruption_training_frequency", "")}

    def _section_corruption_incidents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        incidents = data.get("corruption_incidents", [])
        return {"title": "Corruption and Bribery Incidents", "incident_count": len(incidents),
                "confirmed_cases": sum(1 for i in incidents if i.get("confirmed", False)),
                "fines_eur": data.get("corruption_fines_eur", 0.0),
                "incidents": [{"description": i.get("description", ""), "type": i.get("type", ""),
                               "confirmed": i.get("confirmed", False), "action_taken": i.get("action_taken", "")}
                              for i in incidents]}

    def _section_political_influence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Political Influence and Engagement",
                "political_donations_eur": data.get("political_donations_eur", 0.0),
                "political_donations_prohibited": data.get("political_donations_prohibited", False),
                "trade_association_memberships": data.get("trade_association_memberships", []),
                "policy_engagement_topics": data.get("policy_engagement_topics", [])}

    def _section_lobbying_expenditure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Lobbying Expenditure", "total_lobbying_eur": data.get("lobbying_total_eur", 0.0),
                "registered_in_eu_register": data.get("registered_eu_transparency", False),
                "lobbying_topics": data.get("lobbying_topics", []),
                "lobbying_by_region": data.get("lobbying_by_region", {})}

    def _section_payment_practices(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Payment Practices", "avg_payment_days": data.get("avg_payment_days", 0),
                "late_payments_pct": round(data.get("late_payments_pct", 0.0), 1),
                "standard_payment_terms_days": data.get("standard_payment_terms_days", 30),
                "invoices_paid_on_time_pct": round(data.get("invoices_paid_on_time_pct", 0.0), 1)}

    def _section_sme_payment_terms(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "SME Payment Terms", "sme_avg_payment_days": data.get("sme_avg_payment_days", 0),
                "sme_late_payments_pct": round(data.get("sme_late_payments_pct", 0.0), 1),
                "sme_payment_terms_days": data.get("sme_payment_terms_days", 30),
                "prompt_payment_code_signatory": data.get("prompt_payment_code_signatory", False)}

    def _section_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        targets = data.get("governance_targets", [])
        return {"title": "Business Conduct Targets", "target_count": len(targets),
                "targets": [{"name": t.get("name", ""), "metric": t.get("metric", ""),
                             "target_year": t.get("target_year", ""), "target_value": t.get("target_value", 0.0),
                             "current_value": t.get("current_value", 0.0)} for t in targets]}

    # -- Markdown helpers --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# ESRS G1 Business Conduct Report\n\n**Entity:** {data.get('entity_name', '')}  \n"
                f"**Reporting Year:** {data.get('reporting_year', '')}  \n**Generated:** {ts}  \n"
                f"**Standard:** ESRS G1 Business Conduct")

    def _md_conduct_policies(self, d: Dict[str, Any]) -> str:
        sec = self._section_conduct_policies(d)
        coc = "Yes" if sec["has_code_of_conduct"] else "No"
        wb = "Yes" if sec["whistleblower_protection"] else "No"
        lines = [f"## {sec['title']}\n", f"**Code of Conduct:** {coc}  \n**Whistleblower Protection:** {wb}\n"]
        for p in sec["policies"]:
            vc = "Yes" if p["covers_value_chain"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, Value Chain: {vc})")
        return "\n".join(lines)

    def _md_supplier_mgmt(self, d: Dict[str, Any]) -> str:
        sec = self._section_supplier_management(d)
        coc = "Yes" if sec["supplier_code_of_conduct"] else "No"
        return (f"## {sec['title']}\n\n**Supplier CoC:** {coc}  \n**Screened:** {sec['suppliers_screened_pct']:.1f}%  \n"
                f"**Audited:** {sec['suppliers_audited']}  \n**Terminated:** {sec['terminated_for_violations']}")

    def _md_anti_corruption(self, d: Dict[str, Any]) -> str:
        sec = self._section_anti_corruption_training(d)
        return (f"## {sec['title']}\n\n**Employees Trained:** {sec['employees_trained_pct']:.1f}%  \n"
                f"**Board Trained:** {sec['board_trained_pct']:.1f}%  \n"
                f"**Total Hours:** {sec['training_hours_total']}  \n**Frequency:** {sec['training_frequency']}")

    def _md_corruption_incidents(self, d: Dict[str, Any]) -> str:
        sec = self._section_corruption_incidents(d)
        lines = [f"## {sec['title']}\n", f"**Incidents:** {sec['incident_count']} | **Confirmed:** {sec['confirmed_cases']} "
                 f"| **Fines:** EUR {sec['fines_eur']:,.2f}\n"]
        if sec["incidents"]:
            lines.extend(["| Description | Type | Confirmed | Action |", "|-------------|------|:---------:|--------|"])
            for i in sec["incidents"]:
                conf = "Yes" if i["confirmed"] else "No"
                lines.append(f"| {i['description']} | {i['type']} | {conf} | {i['action_taken']} |")
        return "\n".join(lines)

    def _md_political(self, d: Dict[str, Any]) -> str:
        sec = self._section_political_influence(d)
        prohibited = "Yes" if sec["political_donations_prohibited"] else "No"
        return (f"## {sec['title']}\n\n**Political Donations:** EUR {sec['political_donations_eur']:,.2f}  \n"
                f"**Donations Prohibited:** {prohibited}")

    def _md_lobbying(self, d: Dict[str, Any]) -> str:
        sec = self._section_lobbying_expenditure(d)
        reg = "Yes" if sec["registered_in_eu_register"] else "No"
        return (f"## {sec['title']}\n\n**Total:** EUR {sec['total_lobbying_eur']:,.2f}  \n"
                f"**EU Transparency Register:** {reg}")

    def _md_payment(self, d: Dict[str, Any]) -> str:
        sec = self._section_payment_practices(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Avg Payment Days | {sec['avg_payment_days']} |\n"
                f"| Late Payments % | {sec['late_payments_pct']:.1f}% |\n"
                f"| Standard Terms | {sec['standard_payment_terms_days']} days |\n"
                f"| On-Time % | {sec['invoices_paid_on_time_pct']:.1f}% |")

    def _md_sme_payment(self, d: Dict[str, Any]) -> str:
        sec = self._section_sme_payment_terms(d)
        ppc = "Yes" if sec["prompt_payment_code_signatory"] else "No"
        return (f"## {sec['title']}\n\n**SME Avg Payment Days:** {sec['sme_avg_payment_days']}  \n"
                f"**SME Late %:** {sec['sme_late_payments_pct']:.1f}%  \n"
                f"**Prompt Payment Code:** {ppc}")

    def _md_targets(self, d: Dict[str, Any]) -> str:
        sec = self._section_targets(d)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.extend(["| Target | Metric | Current | Goal | Year |", "|--------|--------|--------:|-----:|-----:|"])
            for t in sec["targets"]:
                lines.append(f"| {t['name']} | {t['metric']} | {t['current_value']:.1f} | {t['target_value']:.1f} | {t['target_year']} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    def _css(self) -> str:
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}.report{max-width:900px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.total{font-weight:bold;background:#e8f5e9}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        return (f"<h1>ESRS G1 Business Conduct Report</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_corruption(self, data: Dict[str, Any]) -> str:
        sec = self._section_corruption_incidents(data)
        rows = "".join(f"<tr><td>{i['description']}</td><td>{i['type']}</td>"
                       f"<td>{'Yes' if i['confirmed'] else 'No'}</td></tr>" for i in sec["incidents"])
        return (f"<h2>{sec['title']}</h2>\n<p>Total: {sec['incident_count']} | Confirmed: {sec['confirmed_cases']}</p>\n"
                f"<table><tr><th>Description</th><th>Type</th><th>Confirmed</th></tr>{rows}</table>")

    def _html_payment(self, data: Dict[str, Any]) -> str:
        sec = self._section_payment_practices(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Avg Days</td><td>{sec['avg_payment_days']}</td></tr>"
                f"<tr><td>Late %</td><td>{sec['late_payments_pct']:.1f}%</td></tr></table>")

    def _html_lobbying(self, data: Dict[str, Any]) -> str:
        sec = self._section_lobbying_expenditure(data)
        return (f"<h2>{sec['title']}</h2>\n"
                f"<p>Total: EUR {sec['total_lobbying_eur']:,.2f}</p>")


# Alias for backward compatibility with templates/__init__.py
G1GovernanceReport = G1GovernanceReportTemplate
