# -*- coding: utf-8 -*-
"""
LabelComplianceReportTemplate - EU Green Claims Label Compliance Assessment

Evaluates eco-labels and environmental certification marks used by an entity
against the EU Green Claims Directive Article 10 requirements. Audits label
schemes for governance quality, third-party verification, transparency, and
certificate validity. Generates per-label compliance findings with a
prioritised remediation plan for non-compliant labels.

Sections:
    1. Executive Summary - Overall label compliance score and key findings
    2. Label Inventory - Full catalogue of eco-labels in use
    3. Scheme Assessment - Label scheme governance evaluation (Article 10)
    4. Certificate Status - Validity and expiry tracking
    5. Compliance Findings - Per-label compliance determination
    6. Remediation Plan - Prioritised corrective actions
    7. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = ["LabelComplianceReportTemplate"]

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "executive_summary", "title": "Executive Summary", "order": 1},
    {"id": "label_inventory", "title": "Label Inventory", "order": 2},
    {"id": "scheme_assessment", "title": "Scheme Assessment", "order": 3},
    {"id": "certificate_status", "title": "Certificate Status", "order": 4},
    {"id": "compliance_findings", "title": "Compliance Findings", "order": 5},
    {"id": "remediation_plan", "title": "Remediation Plan", "order": 6},
    {"id": "provenance", "title": "Provenance", "order": 7},
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

class LabelComplianceReportTemplate:
    """
    EU Green Claims Directive - Label Compliance Assessment Report.

    Audits all eco-labels and environmental certification marks used by an
    entity against EU Green Claims Directive Article 10 requirements.
    Evaluates scheme governance (third-party verification, transparency,
    independence), tracks certificate validity, identifies non-compliant
    labels, and produces a prioritised remediation plan.

    Example:
        >>> tpl = LabelComplianceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LabelComplianceReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render label compliance report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_label_inventory(data),
            self._md_scheme_assessment(data),
            self._md_certificate_status(data),
            self._md_compliance_findings(data),
            self._md_remediation_plan(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render label compliance report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_label_inventory(data),
            self._html_scheme_assessment(data),
            self._html_certificate_status(data),
            self._html_compliance_findings(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Label Compliance Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render label compliance report as structured JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "label_compliance_report",
            "directive_reference": "EU Green Claims Directive 2023/0085 Article 10",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "executive_summary": self._section_executive_summary(data),
            "label_inventory": self._section_label_inventory(data),
            "scheme_assessment": self._section_scheme_assessment(data),
            "certificate_status": self._section_certificate_status(data),
            "compliance_findings": self._section_compliance_findings(data),
            "remediation_plan": self._section_remediation_plan(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_sections(self) -> List[Dict[str, Any]]:
        """Return list of available section definitions."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("labels"):
            errors.append("labels list is required")
        if not data.get("reporting_period"):
            warnings.append("reporting_period missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        labels = data.get("labels", [])
        total = len(labels)
        compliant = sum(1 for lb in labels if lb.get("compliant", False))
        expired = sum(1 for lb in labels if lb.get("certificate_expired", False))
        score = round(compliant / total * 100, 1) if total > 0 else 0.0
        return {
            "title": "Executive Summary",
            "total_labels": total,
            "compliant_labels": compliant,
            "non_compliant_labels": total - compliant,
            "expired_certificates": expired,
            "overall_score_pct": score,
            "assessment_status": self._get_status(score),
            "assessment_date": data.get("assessment_date", utcnow().isoformat()),
        }

    def _section_label_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build label inventory section."""
        labels = data.get("labels", [])
        return {
            "title": "Label Inventory",
            "total_labels": len(labels),
            "labels": [
                {
                    "label_id": lb.get("label_id", ""),
                    "label_name": lb.get("label_name", ""),
                    "scheme_owner": lb.get("scheme_owner", ""),
                    "label_type": lb.get("label_type", ""),
                    "iso_type": lb.get("iso_type", ""),
                    "products_covered": lb.get("products_covered", []),
                    "eu_recognised": lb.get("eu_recognised", False),
                }
                for lb in labels
            ],
        }

    def _section_scheme_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scheme assessment section per Article 10."""
        labels = data.get("labels", [])
        assessments = []
        for lb in labels:
            gov = lb.get("governance", {})
            assessments.append({
                "label_id": lb.get("label_id", ""),
                "label_name": lb.get("label_name", ""),
                "independent_verification": gov.get("independent_verification", False),
                "transparent_criteria": gov.get("transparent_criteria", False),
                "publicly_accessible": gov.get("publicly_accessible", False),
                "complaints_mechanism": gov.get("complaints_mechanism", False),
                "periodic_review": gov.get("periodic_review", False),
                "scientific_basis": gov.get("scientific_basis", False),
                "governance_score": gov.get("governance_score", 0.0),
            })
        passing = sum(1 for a in assessments if a["governance_score"] >= 70.0)
        return {
            "title": "Scheme Assessment (Article 10)",
            "total_assessed": len(assessments),
            "passing": passing,
            "failing": len(assessments) - passing,
            "assessments": assessments,
        }

    def _section_certificate_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build certificate status section."""
        labels = data.get("labels", [])
        active = [lb for lb in labels if lb.get("certificate_active", False)]
        expired = [lb for lb in labels if lb.get("certificate_expired", False)]
        expiring = [lb for lb in labels if lb.get("certificate_expiring_soon", False)]
        return {
            "title": "Certificate Status",
            "active_count": len(active),
            "expired_count": len(expired),
            "expiring_soon_count": len(expiring),
            "expired_certificates": [
                {"label_id": lb.get("label_id", ""), "label_name": lb.get("label_name", ""),
                 "expiry_date": lb.get("certificate_expiry", "")}
                for lb in expired
            ],
            "expiring_soon": [
                {"label_id": lb.get("label_id", ""), "label_name": lb.get("label_name", ""),
                 "expiry_date": lb.get("certificate_expiry", "")}
                for lb in expiring
            ],
        }

    def _section_compliance_findings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build compliance findings section."""
        labels = data.get("labels", [])
        compliant = sum(1 for lb in labels if lb.get("compliant", False))
        return {
            "title": "Compliance Findings",
            "total_labels": len(labels),
            "compliant": compliant,
            "non_compliant": len(labels) - compliant,
            "compliance_rate_pct": round(
                compliant / len(labels) * 100, 1
            ) if labels else 0.0,
            "findings": [
                {
                    "label_id": lb.get("label_id", ""),
                    "label_name": lb.get("label_name", ""),
                    "compliant": lb.get("compliant", False),
                    "issues": lb.get("compliance_issues", []),
                    "article_references": lb.get("article_references", []),
                    "severity": lb.get("severity", "low"),
                }
                for lb in labels
            ],
        }

    def _section_remediation_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remediation plan section."""
        actions = data.get("remediation_actions", [])
        return {
            "title": "Remediation Plan",
            "total_actions": len(actions),
            "actions": [
                {
                    "priority": a.get("priority", 0),
                    "label_id": a.get("label_id", ""),
                    "action": a.get("action", ""),
                    "effort": a.get("effort", ""),
                    "deadline": a.get("deadline", ""),
                    "responsible": a.get("responsible", ""),
                    "expected_outcome": a.get("expected_outcome", ""),
                }
                for a in sorted(actions, key=lambda x: x.get("priority", 999))
            ],
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Label Compliance Report - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** EU Green Claims Directive 2023/0085 Article 10"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"### Overall Score: {sec['overall_score_pct']:.1f}%\n\n"
            f"**Status:** {sec['assessment_status']}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Labels | {sec['total_labels']} |\n"
            f"| Compliant | {sec['compliant_labels']} |\n"
            f"| Non-Compliant | {sec['non_compliant_labels']} |\n"
            f"| Expired Certificates | {sec['expired_certificates']} |"
        )

    def _md_label_inventory(self, data: Dict[str, Any]) -> str:
        """Render label inventory as markdown."""
        sec = self._section_label_inventory(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Labels:** {sec['total_labels']}\n",
            "| ID | Label Name | Type | ISO Type | Scheme Owner | EU Recognised |",
            "|----|------------|------|----------|-------------|:------------:|",
        ]
        for lb in sec["labels"]:
            eu = "Yes" if lb["eu_recognised"] else "No"
            lines.append(
                f"| {lb['label_id']} | {lb['label_name']} | {lb['label_type']} "
                f"| {lb['iso_type']} | {lb['scheme_owner']} | {eu} |"
            )
        return "\n".join(lines)

    def _md_scheme_assessment(self, data: Dict[str, Any]) -> str:
        """Render scheme assessment as markdown."""
        sec = self._section_scheme_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Passing:** {sec['passing']} | **Failing:** {sec['failing']}\n",
            "| Label | Indep. | Transparent | Public | Complaints | Science | Score |",
            "|-------|:------:|:-----------:|:------:|:----------:|:-------:|------:|",
        ]
        for a in sec["assessments"]:
            iv = "Yes" if a["independent_verification"] else "No"
            tc = "Yes" if a["transparent_criteria"] else "No"
            pa = "Yes" if a["publicly_accessible"] else "No"
            cm = "Yes" if a["complaints_mechanism"] else "No"
            sb = "Yes" if a["scientific_basis"] else "No"
            lines.append(
                f"| {a['label_name']} | {iv} | {tc} | {pa} | {cm} "
                f"| {sb} | {a['governance_score']:.1f} |"
            )
        return "\n".join(lines)

    def _md_certificate_status(self, data: Dict[str, Any]) -> str:
        """Render certificate status as markdown."""
        sec = self._section_certificate_status(data)
        lines = [
            "## Certificate Status\n",
            f"- **Active:** {sec['active_count']}",
            f"- **Expired:** {sec['expired_count']}",
            f"- **Expiring Soon:** {sec['expiring_soon_count']}\n",
        ]
        if sec["expired_certificates"]:
            lines.append("### Expired Certificates\n")
            for c in sec["expired_certificates"]:
                lines.append(f"- **{c['label_id']}**: {c['label_name']} (expired {c['expiry_date']})")
        if sec["expiring_soon"]:
            lines.append("\n### Expiring Soon\n")
            for c in sec["expiring_soon"]:
                lines.append(f"- **{c['label_id']}**: {c['label_name']} (expires {c['expiry_date']})")
        return "\n".join(lines)

    def _md_compliance_findings(self, data: Dict[str, Any]) -> str:
        """Render compliance findings as markdown."""
        sec = self._section_compliance_findings(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Compliance Rate:** {sec['compliance_rate_pct']:.1f}%  \n"
            f"**Compliant:** {sec['compliant']} | **Non-Compliant:** {sec['non_compliant']}\n",
            "| Label | Status | Severity | Issues |",
            "|-------|--------|----------|--------|",
        ]
        for f in sec["findings"]:
            status = "Compliant" if f["compliant"] else "Non-Compliant"
            issues = "; ".join(f["issues"][:3]) if f["issues"] else "None"
            lines.append(f"| {f['label_name']} | {status} | {f['severity']} | {issues} |")
        return "\n".join(lines)

    def _md_remediation_plan(self, data: Dict[str, Any]) -> str:
        """Render remediation plan as markdown."""
        sec = self._section_remediation_plan(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_actions']}\n",
            "| Priority | Label | Action | Effort | Deadline | Responsible |",
            "|---------:|-------|--------|--------|----------|-------------|",
        ]
        for a in sec["actions"]:
            lines.append(
                f"| {a['priority']} | {a['label_id']} | {a['action']} "
                f"| {a['effort']} | {a['deadline']} | {a['responsible']} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** {self.generated_at.isoformat() if self.generated_at else ''}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-018 EU Green Claims Prep Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1b5e20;border-bottom:2px solid #1b5e20;padding-bottom:.3em}"
            "h2{color:#2e7d32;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8f5e9}"
            ".non-compliant{color:#c62828;font-weight:bold}"
            ".compliant{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Label Compliance Report - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = "compliant" if sec["overall_score_pct"] >= 80.0 else "non-compliant"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Overall Score: {sec['overall_score_pct']:.1f}%</p>\n"
            f"<p><strong>Status:</strong> {sec['assessment_status']}</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Total Labels</td><td>{sec['total_labels']}</td></tr>"
            f"<tr><td>Compliant</td><td>{sec['compliant_labels']}</td></tr>"
            f"<tr><td>Non-Compliant</td><td>{sec['non_compliant_labels']}</td></tr>"
            f"<tr><td>Expired</td><td>{sec['expired_certificates']}</td></tr></table>"
        )

    def _html_label_inventory(self, data: Dict[str, Any]) -> str:
        """Render label inventory HTML."""
        sec = self._section_label_inventory(data)
        rows = "".join(
            f"<tr><td>{lb['label_id']}</td><td>{lb['label_name']}</td>"
            f"<td>{lb['label_type']}</td><td>{lb['scheme_owner']}</td></tr>"
            for lb in sec["labels"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Labels: {sec['total_labels']}</p>\n"
            f"<table><tr><th>ID</th><th>Name</th><th>Type</th><th>Owner</th></tr>"
            f"{rows}</table>"
        )

    def _html_scheme_assessment(self, data: Dict[str, Any]) -> str:
        """Render scheme assessment HTML."""
        sec = self._section_scheme_assessment(data)
        rows = "".join(
            f"<tr><td>{a['label_name']}</td><td>{a['governance_score']:.1f}</td>"
            f"<td>{'Yes' if a['independent_verification'] else 'No'}</td>"
            f"<td>{'Yes' if a['scientific_basis'] else 'No'}</td></tr>"
            for a in sec["assessments"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Passing: {sec['passing']} | Failing: {sec['failing']}</p>\n"
            f"<table><tr><th>Label</th><th>Score</th><th>Independent</th>"
            f"<th>Science</th></tr>{rows}</table>"
        )

    def _html_certificate_status(self, data: Dict[str, Any]) -> str:
        """Render certificate status HTML."""
        sec = self._section_certificate_status(data)
        return (
            f"<h2>Certificate Status</h2>\n"
            f"<ul><li class='compliant'>Active: {sec['active_count']}</li>"
            f"<li class='non-compliant'>Expired: {sec['expired_count']}</li>"
            f"<li>Expiring Soon: {sec['expiring_soon_count']}</li></ul>"
        )

    def _html_compliance_findings(self, data: Dict[str, Any]) -> str:
        """Render compliance findings HTML."""
        sec = self._section_compliance_findings(data)
        rows = "".join(
            f"<tr><td>{f['label_name']}</td>"
            f"<td class='{'compliant' if f['compliant'] else 'non-compliant'}'>"
            f"{'Compliant' if f['compliant'] else 'Non-Compliant'}</td>"
            f"<td>{f['severity']}</td></tr>"
            for f in sec["findings"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Compliance Rate: {sec['compliance_rate_pct']:.1f}%</p>\n"
            f"<table><tr><th>Label</th><th>Status</th><th>Severity</th></tr>"
            f"{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_status(self, score: float) -> str:
        """Determine assessment status from overall score."""
        if score >= 95.0:
            return "Fully Compliant"
        elif score >= 80.0:
            return "Largely Compliant (Minor Gaps)"
        elif score >= 60.0:
            return "Partially Compliant (Action Required)"
        elif score >= 40.0:
            return "Non-Compliant (Significant Action Required)"
        else:
            return "Critical Non-Compliance"
