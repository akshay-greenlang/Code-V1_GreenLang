# -*- coding: utf-8 -*-
"""
RegulatorySubmissionReportTemplate - EU Green Claims CAB Submission Package

Formats a complete regulatory submission package for Conformity Assessment
Body (CAB) review under the EU Green Claims Directive. Assembles cover page,
substantiation summary, evidence chain, lifecycle assessment references,
label compliance status, trader declaration, and full audit trail into a
structured document ready for CAB submission.

Sections:
    1. Cover Page - Submission metadata and entity identification
    2. Substantiation Summary - Claim substantiation overview
    3. Evidence Chain - Linked evidence with provenance
    4. Lifecycle Assessment - LCA/PEF references and results
    5. Label Compliance - Eco-label compliance status
    6. Trader Declaration - Formal compliance declaration
    7. Audit Trail - Complete processing audit log
    8. Provenance - Data lineage and hash chain

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

__all__ = ["RegulatorySubmissionReportTemplate"]

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "cover_page", "title": "Cover Page", "order": 1},
    {"id": "substantiation_summary", "title": "Substantiation Summary", "order": 2},
    {"id": "evidence_chain", "title": "Evidence Chain", "order": 3},
    {"id": "lifecycle_assessment", "title": "Lifecycle Assessment", "order": 4},
    {"id": "label_compliance", "title": "Label Compliance", "order": 5},
    {"id": "trader_declaration", "title": "Trader Declaration", "order": 6},
    {"id": "audit_trail", "title": "Audit Trail", "order": 7},
    {"id": "provenance", "title": "Provenance", "order": 8},
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

class RegulatorySubmissionReportTemplate:
    """
    EU Green Claims Directive - CAB Regulatory Submission Package.

    Formats a complete submission package for Conformity Assessment Body
    (CAB) review assembling cover page, substantiation summary, evidence
    chain with provenance links, lifecycle assessment references, label
    compliance status, trader declaration, and a complete audit trail.
    Includes SHA-256 provenance hashing for document integrity.

    Example:
        >>> tpl = RegulatorySubmissionReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatorySubmissionReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render regulatory submission report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_cover_page(data),
            self._md_substantiation_summary(data),
            self._md_evidence_chain(data),
            self._md_lifecycle_assessment(data),
            self._md_label_compliance(data),
            self._md_trader_declaration(data),
            self._md_audit_trail(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory submission report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_cover_page(data),
            self._html_substantiation_summary(data),
            self._html_evidence_chain(data),
            self._html_lifecycle_assessment(data),
            self._html_label_compliance(data),
            self._html_trader_declaration(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulatory Submission - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory submission report as structured JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_submission_report",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "submission_id": data.get("submission_id", ""),
            "cover_page": self._section_cover_page(data),
            "substantiation_summary": self._section_substantiation_summary(data),
            "evidence_chain": self._section_evidence_chain(data),
            "lifecycle_assessment": self._section_lifecycle_assessment(data),
            "label_compliance": self._section_label_compliance(data),
            "trader_declaration": self._section_trader_declaration(data),
            "audit_trail": self._section_audit_trail(data),
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
        if not data.get("claims"):
            errors.append("claims list is required for submission")
        if not data.get("submission_id"):
            warnings.append("submission_id missing; will be auto-generated")
        if not data.get("trader"):
            warnings.append("trader details missing; declaration will be incomplete")
        if not data.get("evidence_items"):
            warnings.append("evidence_items missing; evidence chain will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_cover_page(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cover page section."""
        claims = data.get("claims", [])
        return {
            "title": "Cover Page",
            "submission_id": data.get("submission_id", ""),
            "entity_name": data.get("entity_name", ""),
            "entity_registration": data.get("entity_registration", ""),
            "entity_address": data.get("entity_address", ""),
            "member_state": data.get("member_state", ""),
            "competent_authority": data.get("competent_authority", ""),
            "cab_name": data.get("cab_name", ""),
            "cab_accreditation": data.get("cab_accreditation", ""),
            "submission_date": data.get(
                "submission_date", utcnow().isoformat()
            ),
            "submission_type": data.get("submission_type", "initial"),
            "total_claims": len(claims),
            "contact_name": data.get("contact_name", ""),
            "contact_email": data.get("contact_email", ""),
        }

    def _section_substantiation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build substantiation summary section."""
        claims = data.get("claims", [])
        substantiated = sum(
            1 for c in claims if c.get("substantiated", False)
        )
        return {
            "title": "Substantiation Summary",
            "total_claims": len(claims),
            "substantiated": substantiated,
            "unsubstantiated": len(claims) - substantiated,
            "substantiation_rate_pct": round(
                substantiated / len(claims) * 100, 1
            ) if claims else 0.0,
            "claims": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c.get("claim_text", ""),
                    "claim_type": c.get("claim_type", ""),
                    "product_or_service": c.get("product_or_service", ""),
                    "substantiated": c.get("substantiated", False),
                    "methodology": c.get("methodology", ""),
                    "evidence_refs": c.get("evidence_refs", []),
                }
                for c in claims
            ],
        }

    def _section_evidence_chain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build evidence chain section."""
        evidence = data.get("evidence_items", [])
        return {
            "title": "Evidence Chain",
            "total_items": len(evidence),
            "items": [
                {
                    "evidence_id": e.get("evidence_id", ""),
                    "title": e.get("title", ""),
                    "doc_type": e.get("doc_type", ""),
                    "source": e.get("source", ""),
                    "date_issued": e.get("date_issued", ""),
                    "valid_until": e.get("valid_until", ""),
                    "related_claims": e.get("related_claims", []),
                    "file_reference": e.get("file_reference", ""),
                    "file_hash": e.get("file_hash", ""),
                    "chain_position": e.get("chain_position", 0),
                }
                for e in sorted(
                    evidence,
                    key=lambda x: x.get("chain_position", 0),
                )
            ],
            "chain_integrity_hash": _compute_hash(evidence),
        }

    def _section_lifecycle_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lifecycle assessment section."""
        lca = data.get("lifecycle_assessment", {})
        return {
            "title": "Lifecycle Assessment",
            "lca_available": bool(lca),
            "lca_type": lca.get("type", ""),
            "methodology": lca.get("methodology", ""),
            "functional_unit": lca.get("functional_unit", ""),
            "system_boundary": lca.get("system_boundary", ""),
            "pef_score": lca.get("pef_score", 0.0),
            "pef_unit": lca.get("pef_unit", "mPt"),
            "data_quality_rating": lca.get("data_quality_rating", ""),
            "lca_report_ref": lca.get("report_reference", ""),
            "lca_report_hash": lca.get("report_hash", ""),
            "critical_review": lca.get("critical_review", False),
            "reviewer_name": lca.get("reviewer_name", ""),
        }

    def _section_label_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build label compliance section."""
        labels = data.get("labels", [])
        compliant = sum(1 for lb in labels if lb.get("compliant", False))
        return {
            "title": "Label Compliance",
            "total_labels": len(labels),
            "compliant": compliant,
            "non_compliant": len(labels) - compliant,
            "compliance_rate_pct": round(
                compliant / len(labels) * 100, 1
            ) if labels else 0.0,
            "labels": [
                {
                    "label_id": lb.get("label_id", ""),
                    "label_name": lb.get("label_name", ""),
                    "compliant": lb.get("compliant", False),
                    "certificate_ref": lb.get("certificate_ref", ""),
                    "valid_until": lb.get("valid_until", ""),
                }
                for lb in labels
            ],
        }

    def _section_trader_declaration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build trader declaration section."""
        trader = data.get("trader", {})
        return {
            "title": "Trader Declaration",
            "declaration_date": data.get(
                "declaration_date", utcnow().isoformat()
            ),
            "trader_name": trader.get("name", ""),
            "trader_title": trader.get("title", ""),
            "trader_email": trader.get("email", ""),
            "company_name": data.get("entity_name", ""),
            "declaration_text": data.get(
                "declaration_text",
                (
                    "I, the undersigned trader, hereby declare that all "
                    "environmental claims included in this submission have "
                    "been substantiated in accordance with the requirements "
                    "of the EU Green Claims Directive (2023/0085). All "
                    "supporting evidence is accurate, current, and available "
                    "for review by the designated Conformity Assessment Body."
                ),
            ),
            "data_accuracy_confirmed": data.get(
                "data_accuracy_confirmed", False
            ),
            "terms_accepted": data.get("terms_accepted", False),
        }

    def _section_audit_trail(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build audit trail section."""
        audit_entries = data.get("audit_trail", [])
        return {
            "title": "Audit Trail",
            "total_entries": len(audit_entries),
            "entries": [
                {
                    "timestamp": e.get("timestamp", ""),
                    "action": e.get("action", ""),
                    "actor": e.get("actor", ""),
                    "component": e.get("component", ""),
                    "details": e.get("details", ""),
                    "hash": e.get("hash", ""),
                }
                for e in sorted(
                    audit_entries,
                    key=lambda x: x.get("timestamp", ""),
                )
            ],
            "trail_integrity_hash": _compute_hash(audit_entries),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_cover_page(self, data: Dict[str, Any]) -> str:
        """Render cover page as markdown."""
        sec = self._section_cover_page(data)
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Regulatory Submission - EU Green Claims Directive\n\n"
            f"**CONFIDENTIAL - CAB SUBMISSION PACKAGE**\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Submission ID | {sec['submission_id']} |\n"
            f"| Entity | {sec['entity_name']} |\n"
            f"| Registration | {sec['entity_registration']} |\n"
            f"| Address | {sec['entity_address']} |\n"
            f"| Member State | {sec['member_state']} |\n"
            f"| Competent Authority | {sec['competent_authority']} |\n"
            f"| CAB | {sec['cab_name']} |\n"
            f"| CAB Accreditation | {sec['cab_accreditation']} |\n"
            f"| Submission Date | {sec['submission_date']} |\n"
            f"| Submission Type | {sec['submission_type']} |\n"
            f"| Total Claims | {sec['total_claims']} |\n"
            f"| Contact | {sec['contact_name']} ({sec['contact_email']}) |\n"
            f"| Generated | {ts} |"
        )

    def _md_substantiation_summary(self, data: Dict[str, Any]) -> str:
        """Render substantiation summary as markdown."""
        sec = self._section_substantiation_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Substantiation Rate:** {sec['substantiation_rate_pct']:.1f}%  \n"
            f"**Substantiated:** {sec['substantiated']}/{sec['total_claims']}  \n"
            f"**Unsubstantiated:** {sec['unsubstantiated']}\n",
            "| ID | Claim | Type | Product | Substantiated | Methodology |",
            "|----|-------|------|---------|:------------:|-------------|",
        ]
        for c in sec["claims"]:
            sub = "Yes" if c["substantiated"] else "No"
            lines.append(
                f"| {c['claim_id']} | {c['claim_text'][:45]} "
                f"| {c['claim_type']} | {c['product_or_service']} "
                f"| {sub} | {c['methodology']} |"
            )
        return "\n".join(lines)

    def _md_evidence_chain(self, data: Dict[str, Any]) -> str:
        """Render evidence chain as markdown."""
        sec = self._section_evidence_chain(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Items:** {sec['total_items']}  \n"
            f"**Chain Integrity Hash:** `{sec['chain_integrity_hash'][:16]}...`\n",
            "| # | ID | Title | Type | Source | Issued | Valid Until | Hash |",
            "|--:|----|-------|------|--------|--------|------------|------|",
        ]
        for e in sec["items"]:
            short_hash = e["file_hash"][:12] + "..." if e["file_hash"] else "N/A"
            lines.append(
                f"| {e['chain_position']} | {e['evidence_id']} "
                f"| {e['title'][:35]} | {e['doc_type']} "
                f"| {e['source']} | {e['date_issued']} "
                f"| {e['valid_until']} | `{short_hash}` |"
            )
        return "\n".join(lines)

    def _md_lifecycle_assessment(self, data: Dict[str, Any]) -> str:
        """Render lifecycle assessment as markdown."""
        sec = self._section_lifecycle_assessment(data)
        if not sec["lca_available"]:
            return (
                "## Lifecycle Assessment\n\n"
                "No lifecycle assessment data provided for this submission."
            )
        review = "Yes" if sec["critical_review"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"- **Type:** {sec['lca_type']}\n"
            f"- **Methodology:** {sec['methodology']}\n"
            f"- **Functional Unit:** {sec['functional_unit']}\n"
            f"- **System Boundary:** {sec['system_boundary']}\n"
            f"- **PEF Score:** {sec['pef_score']:.2f} {sec['pef_unit']}\n"
            f"- **Data Quality:** {sec['data_quality_rating']}\n"
            f"- **Critical Review:** {review}\n"
            f"- **Reviewer:** {sec['reviewer_name']}\n"
            f"- **Report Reference:** {sec['lca_report_ref']}"
        )

    def _md_label_compliance(self, data: Dict[str, Any]) -> str:
        """Render label compliance as markdown."""
        sec = self._section_label_compliance(data)
        if not sec["labels"]:
            return (
                "## Label Compliance\n\n"
                "No eco-labels applicable to this submission."
            )
        lines = [
            f"## {sec['title']}\n",
            f"**Compliance Rate:** {sec['compliance_rate_pct']:.1f}%  \n"
            f"**Compliant:** {sec['compliant']}/{sec['total_labels']}\n",
            "| ID | Label | Compliant | Certificate | Valid Until |",
            "|----|-------|:---------:|-------------|------------|",
        ]
        for lb in sec["labels"]:
            comp = "Yes" if lb["compliant"] else "No"
            lines.append(
                f"| {lb['label_id']} | {lb['label_name']} | {comp} "
                f"| {lb['certificate_ref']} | {lb['valid_until']} |"
            )
        return "\n".join(lines)

    def _md_trader_declaration(self, data: Dict[str, Any]) -> str:
        """Render trader declaration as markdown."""
        sec = self._section_trader_declaration(data)
        accuracy = "Yes" if sec["data_accuracy_confirmed"] else "No"
        terms = "Yes" if sec["terms_accepted"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"> {sec['declaration_text']}\n\n"
            f"**Trader Name:** {sec['trader_name']}  \n"
            f"**Title:** {sec['trader_title']}  \n"
            f"**Company:** {sec['company_name']}  \n"
            f"**Date:** {sec['declaration_date']}  \n"
            f"**Data Accuracy Confirmed:** {accuracy}  \n"
            f"**Terms Accepted:** {terms}"
        )

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        """Render audit trail as markdown."""
        sec = self._section_audit_trail(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Entries:** {sec['total_entries']}  \n"
            f"**Trail Integrity Hash:** "
            f"`{sec['trail_integrity_hash'][:16]}...`\n",
            "| Timestamp | Action | Actor | Component | Hash |",
            "|-----------|--------|-------|-----------|------|",
        ]
        for e in sec["entries"]:
            short_hash = e["hash"][:12] + "..." if e["hash"] else "N/A"
            lines.append(
                f"| {e['timestamp']} | {e['action']} "
                f"| {e['actor']} | {e['component']} | `{short_hash}` |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** "
            f"{self.generated_at.isoformat() if self.generated_at else ''}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Submission package generated by PACK-018 "
            f"EU Green Claims Prep Pack on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".verified{color:#2e7d32;font-weight:bold}"
            ".pending{color:#e65100;font-weight:bold}"
            ".failed{color:#c62828;font-weight:bold}"
            "blockquote{border-left:4px solid #0d47a1;margin:1em 0;"
            "padding:0.5em 1em;background:#f5f5f5}"
        )

    def _html_cover_page(self, data: Dict[str, Any]) -> str:
        """Render cover page HTML."""
        sec = self._section_cover_page(data)
        return (
            f"<h1>Regulatory Submission - EU Green Claims Directive</h1>\n"
            f"<p><strong>CONFIDENTIAL - CAB SUBMISSION PACKAGE</strong></p>\n"
            f"<table>"
            f"<tr><td><strong>Entity</strong></td>"
            f"<td>{sec['entity_name']}</td></tr>"
            f"<tr><td><strong>Submission ID</strong></td>"
            f"<td>{sec['submission_id']}</td></tr>"
            f"<tr><td><strong>Member State</strong></td>"
            f"<td>{sec['member_state']}</td></tr>"
            f"<tr><td><strong>CAB</strong></td>"
            f"<td>{sec['cab_name']}</td></tr>"
            f"<tr><td><strong>Claims</strong></td>"
            f"<td>{sec['total_claims']}</td></tr>"
            f"<tr><td><strong>Type</strong></td>"
            f"<td>{sec['submission_type']}</td></tr>"
            f"</table>"
        )

    def _html_substantiation_summary(self, data: Dict[str, Any]) -> str:
        """Render substantiation summary HTML."""
        sec = self._section_substantiation_summary(data)
        rows = "".join(
            f"<tr><td>{c['claim_id']}</td><td>{c['claim_text'][:45]}</td>"
            f"<td>{c['claim_type']}</td>"
            f"<td>{'Yes' if c['substantiated'] else 'No'}</td></tr>"
            for c in sec["claims"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Rate: {sec['substantiation_rate_pct']:.1f}% | "
            f"Substantiated: {sec['substantiated']}/{sec['total_claims']}</p>\n"
            f"<table><tr><th>ID</th><th>Claim</th><th>Type</th>"
            f"<th>Substantiated</th></tr>{rows}</table>"
        )

    def _html_evidence_chain(self, data: Dict[str, Any]) -> str:
        """Render evidence chain HTML."""
        sec = self._section_evidence_chain(data)
        rows = "".join(
            f"<tr><td>{e['chain_position']}</td>"
            f"<td>{e['evidence_id']}</td>"
            f"<td>{e['title'][:35]}</td>"
            f"<td>{e['doc_type']}</td>"
            f"<td>{e['source']}</td></tr>"
            for e in sec["items"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Items: {sec['total_items']}</p>\n"
            f"<table><tr><th>#</th><th>ID</th><th>Title</th>"
            f"<th>Type</th><th>Source</th></tr>{rows}</table>"
        )

    def _html_lifecycle_assessment(self, data: Dict[str, Any]) -> str:
        """Render lifecycle assessment HTML."""
        sec = self._section_lifecycle_assessment(data)
        if not sec["lca_available"]:
            return "<h2>Lifecycle Assessment</h2>\n<p>Not applicable.</p>"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table>"
            f"<tr><td><strong>Type</strong></td>"
            f"<td>{sec['lca_type']}</td></tr>"
            f"<tr><td><strong>PEF Score</strong></td>"
            f"<td>{sec['pef_score']:.2f} {sec['pef_unit']}</td></tr>"
            f"<tr><td><strong>Data Quality</strong></td>"
            f"<td>{sec['data_quality_rating']}</td></tr>"
            f"<tr><td><strong>Critical Review</strong></td>"
            f"<td>{'Yes' if sec['critical_review'] else 'No'}</td></tr>"
            f"</table>"
        )

    def _html_label_compliance(self, data: Dict[str, Any]) -> str:
        """Render label compliance HTML."""
        sec = self._section_label_compliance(data)
        rows = "".join(
            f"<tr><td>{lb['label_id']}</td><td>{lb['label_name']}</td>"
            f"<td>{'Yes' if lb['compliant'] else 'No'}</td>"
            f"<td>{lb['valid_until']}</td></tr>"
            for lb in sec["labels"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Compliance Rate: {sec['compliance_rate_pct']:.1f}%</p>\n"
            f"<table><tr><th>ID</th><th>Label</th><th>Compliant</th>"
            f"<th>Valid Until</th></tr>{rows}</table>"
        )

    def _html_trader_declaration(self, data: Dict[str, Any]) -> str:
        """Render trader declaration HTML."""
        sec = self._section_trader_declaration(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<blockquote>{sec['declaration_text']}</blockquote>\n"
            f"<p><strong>Trader:</strong> {sec['trader_name']} "
            f"({sec['trader_title']})</p>\n"
            f"<p><strong>Company:</strong> {sec['company_name']}</p>\n"
            f"<p><strong>Date:</strong> {sec['declaration_date']}</p>"
        )
