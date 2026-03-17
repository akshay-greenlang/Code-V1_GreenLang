# -*- coding: utf-8 -*-
"""
EvidenceDossierReportTemplate - EU Green Claims Evidence Dossier

Compiles all supporting evidence for environmental claims into a
verifier-ready dossier. Tracks document inventory, chain of custody,
validity windows, and completeness assessment for each evidence item
in accordance with EU Green Claims Directive substantiation requirements.

Sections:
    1. Dossier Overview - Summary of evidence package
    2. Document Inventory - All documents with metadata
    3. Chain of Custody - Evidence provenance and handling
    4. Validity Status - Currency and expiry tracking
    5. Completeness Assessment - Gap identification per claim
    6. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "dossier_overview", "title": "Dossier Overview", "order": 1},
    {"id": "document_inventory", "title": "Document Inventory", "order": 2},
    {"id": "chain_of_custody", "title": "Chain of Custody", "order": 3},
    {"id": "validity_status", "title": "Validity Status", "order": 4},
    {"id": "completeness_assessment", "title": "Completeness Assessment", "order": 5},
    {"id": "provenance", "title": "Provenance", "order": 6},
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


class EvidenceDossierReportTemplate:
    """
    EU Green Claims Directive - Evidence Dossier Report.

    Compiles and organises all evidence supporting environmental claims
    into a structured, verifier-ready package. Tracks document provenance,
    validity periods, chain of custody, and identifies completeness gaps
    against EU Green Claims Directive requirements.

    Example:
        >>> tpl = EvidenceDossierReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EvidenceDossierReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render evidence dossier report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_dossier_overview(data),
            self._md_document_inventory(data),
            self._md_chain_of_custody(data),
            self._md_validity_status(data),
            self._md_completeness_assessment(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render evidence dossier report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_dossier_overview(data),
            self._html_document_inventory(data),
            self._html_validity_status(data),
            self._html_completeness(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Evidence Dossier Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render evidence dossier report as structured JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "evidence_dossier_report",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "dossier_overview": self._section_dossier_overview(data),
            "document_inventory": self._section_document_inventory(data),
            "chain_of_custody": self._section_chain_of_custody(data),
            "validity_status": self._section_validity_status(data),
            "completeness_assessment": self._section_completeness(data),
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
        if not data.get("documents"):
            errors.append("documents list is required")
        if not data.get("claims"):
            warnings.append("claims list missing; completeness check will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_dossier_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build dossier overview section."""
        documents = data.get("documents", [])
        claims = data.get("claims", [])
        valid_docs = sum(1 for d in documents if d.get("valid", False))
        expired_docs = sum(1 for d in documents if d.get("expired", False))
        return {
            "title": "Dossier Overview",
            "total_documents": len(documents),
            "valid_documents": valid_docs,
            "expired_documents": expired_docs,
            "claims_covered": len(claims),
            "completeness_pct": round(
                valid_docs / len(documents) * 100, 1
            ) if documents else 0.0,
            "dossier_status": self._get_dossier_status(valid_docs, len(documents)),
            "prepared_by": data.get("prepared_by", ""),
            "preparation_date": data.get("preparation_date", _utcnow().isoformat()),
        }

    def _section_document_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build document inventory section."""
        documents = data.get("documents", [])
        return {
            "title": "Document Inventory",
            "total_documents": len(documents),
            "documents": [
                {
                    "doc_id": d.get("doc_id", ""),
                    "title": d.get("title", ""),
                    "doc_type": d.get("doc_type", ""),
                    "source": d.get("source", ""),
                    "date_issued": d.get("date_issued", ""),
                    "expiry_date": d.get("expiry_date", ""),
                    "related_claims": d.get("related_claims", []),
                    "file_hash": d.get("file_hash", ""),
                }
                for d in documents
            ],
        }

    def _section_chain_of_custody(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chain of custody section."""
        custody_records = data.get("custody_records", [])
        return {
            "title": "Chain of Custody",
            "total_records": len(custody_records),
            "records": [
                {
                    "doc_id": r.get("doc_id", ""),
                    "action": r.get("action", ""),
                    "actor": r.get("actor", ""),
                    "timestamp": r.get("timestamp", ""),
                    "notes": r.get("notes", ""),
                    "integrity_hash": r.get("integrity_hash", ""),
                }
                for r in custody_records
            ],
        }

    def _section_validity_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build validity status section."""
        documents = data.get("documents", [])
        valid = [d for d in documents if d.get("valid", False)]
        expired = [d for d in documents if d.get("expired", False)]
        expiring_soon = [d for d in documents if d.get("expiring_soon", False)]
        return {
            "title": "Validity Status",
            "valid_count": len(valid),
            "expired_count": len(expired),
            "expiring_soon_count": len(expiring_soon),
            "expired_documents": [
                {"doc_id": d.get("doc_id", ""), "title": d.get("title", ""),
                 "expiry_date": d.get("expiry_date", "")}
                for d in expired
            ],
            "expiring_soon_documents": [
                {"doc_id": d.get("doc_id", ""), "title": d.get("title", ""),
                 "expiry_date": d.get("expiry_date", "")}
                for d in expiring_soon
            ],
        }

    def _section_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build completeness assessment section."""
        claims = data.get("claims", [])
        documents = data.get("documents", [])
        doc_claim_map: Dict[str, List[str]] = {}
        for d in documents:
            for cid in d.get("related_claims", []):
                doc_claim_map.setdefault(cid, []).append(d.get("doc_id", ""))
        gaps = []
        for c in claims:
            cid = c.get("claim_id", "")
            supporting_docs = doc_claim_map.get(cid, [])
            required = c.get("required_evidence_count", 1)
            if len(supporting_docs) < required:
                gaps.append({
                    "claim_id": cid,
                    "claim_text": c.get("claim_text", ""),
                    "documents_found": len(supporting_docs),
                    "documents_required": required,
                    "missing_types": c.get("missing_evidence_types", []),
                })
        covered = sum(1 for c in claims if doc_claim_map.get(c.get("claim_id", ""), []))
        return {
            "title": "Completeness Assessment",
            "total_claims": len(claims),
            "claims_with_evidence": covered,
            "claims_without_evidence": len(claims) - covered,
            "completeness_pct": round(covered / len(claims) * 100, 1) if claims else 0.0,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Evidence Dossier Report - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** EU Green Claims Directive 2023/0085"
        )

    def _md_dossier_overview(self, data: Dict[str, Any]) -> str:
        """Render dossier overview as markdown."""
        sec = self._section_dossier_overview(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Dossier Status:** {sec['dossier_status']}  \n"
            f"**Completeness:** {sec['completeness_pct']:.1f}%\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Documents | {sec['total_documents']} |\n"
            f"| Valid | {sec['valid_documents']} |\n"
            f"| Expired | {sec['expired_documents']} |\n"
            f"| Claims Covered | {sec['claims_covered']} |"
        )

    def _md_document_inventory(self, data: Dict[str, Any]) -> str:
        """Render document inventory as markdown."""
        sec = self._section_document_inventory(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Documents:** {sec['total_documents']}\n",
            "| ID | Title | Type | Source | Issued | Expiry |",
            "|----|-------|------|--------|--------|--------|",
        ]
        for d in sec["documents"]:
            lines.append(
                f"| {d['doc_id']} | {d['title'][:50]} | {d['doc_type']} "
                f"| {d['source']} | {d['date_issued']} | {d['expiry_date']} |"
            )
        return "\n".join(lines)

    def _md_chain_of_custody(self, data: Dict[str, Any]) -> str:
        """Render chain of custody as markdown."""
        sec = self._section_chain_of_custody(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Records:** {sec['total_records']}\n",
            "| Doc ID | Action | Actor | Timestamp |",
            "|--------|--------|-------|-----------|",
        ]
        for r in sec["records"]:
            lines.append(
                f"| {r['doc_id']} | {r['action']} | {r['actor']} | {r['timestamp']} |"
            )
        return "\n".join(lines)

    def _md_validity_status(self, data: Dict[str, Any]) -> str:
        """Render validity status as markdown."""
        sec = self._section_validity_status(data)
        lines = [
            "## Validity Status\n",
            f"- **Valid:** {sec['valid_count']}",
            f"- **Expired:** {sec['expired_count']}",
            f"- **Expiring Soon:** {sec['expiring_soon_count']}\n",
        ]
        if sec["expired_documents"]:
            lines.append("### Expired Documents\n")
            for d in sec["expired_documents"]:
                lines.append(f"- **{d['doc_id']}**: {d['title']} (expired {d['expiry_date']})")
        if sec["expiring_soon_documents"]:
            lines.append("\n### Expiring Soon\n")
            for d in sec["expiring_soon_documents"]:
                lines.append(f"- **{d['doc_id']}**: {d['title']} (expires {d['expiry_date']})")
        return "\n".join(lines)

    def _md_completeness_assessment(self, data: Dict[str, Any]) -> str:
        """Render completeness assessment as markdown."""
        sec = self._section_completeness(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Completeness:** {sec['completeness_pct']:.1f}%  \n"
            f"**Claims with Evidence:** {sec['claims_with_evidence']}/{sec['total_claims']}\n",
        ]
        if sec["gaps"]:
            lines.append("### Evidence Gaps\n")
            lines.append("| Claim ID | Found | Required | Missing Types |")
            lines.append("|----------|------:|---------:|---------------|")
            for g in sec["gaps"]:
                missing = ", ".join(g["missing_types"]) if g["missing_types"] else "N/A"
                lines.append(
                    f"| {g['claim_id']} | {g['documents_found']} "
                    f"| {g['documents_required']} | {missing} |"
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
        return f"---\n*Dossier generated by PACK-018 EU Green Claims Prep Pack on {ts}*"

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
            ".expired{color:#c62828;font-weight:bold}"
            ".valid{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Evidence Dossier Report - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_dossier_overview(self, data: Dict[str, Any]) -> str:
        """Render dossier overview HTML."""
        sec = self._section_dossier_overview(data)
        css_class = "valid" if sec["completeness_pct"] >= 80.0 else "expired"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Completeness: {sec['completeness_pct']:.1f}%</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Total Documents</td><td>{sec['total_documents']}</td></tr>"
            f"<tr><td>Valid</td><td>{sec['valid_documents']}</td></tr>"
            f"<tr><td>Expired</td><td>{sec['expired_documents']}</td></tr></table>"
        )

    def _html_document_inventory(self, data: Dict[str, Any]) -> str:
        """Render document inventory HTML."""
        sec = self._section_document_inventory(data)
        rows = "".join(
            f"<tr><td>{d['doc_id']}</td><td>{d['title'][:50]}</td>"
            f"<td>{d['doc_type']}</td><td>{d['source']}</td></tr>"
            for d in sec["documents"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>ID</th><th>Title</th><th>Type</th><th>Source</th></tr>"
            f"{rows}</table>"
        )

    def _html_validity_status(self, data: Dict[str, Any]) -> str:
        """Render validity status HTML."""
        sec = self._section_validity_status(data)
        return (
            f"<h2>Validity Status</h2>\n"
            f"<ul><li class='valid'>Valid: {sec['valid_count']}</li>"
            f"<li class='expired'>Expired: {sec['expired_count']}</li>"
            f"<li>Expiring Soon: {sec['expiring_soon_count']}</li></ul>"
        )

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        """Render completeness assessment HTML."""
        sec = self._section_completeness(data)
        rows = "".join(
            f"<tr><td>{g['claim_id']}</td><td>{g['documents_found']}</td>"
            f"<td>{g['documents_required']}</td></tr>"
            for g in sec["gaps"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Completeness: {sec['completeness_pct']:.1f}%</p>\n"
            f"<table><tr><th>Claim</th><th>Found</th><th>Required</th></tr>"
            f"{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dossier_status(self, valid: int, total: int) -> str:
        """Determine dossier status from document validity counts."""
        if total == 0:
            return "Empty Dossier"
        pct = valid / total * 100
        if pct >= 95.0:
            return "Verifier Ready"
        elif pct >= 80.0:
            return "Near Complete (Minor Gaps)"
        elif pct >= 60.0:
            return "Partially Complete (Action Required)"
        else:
            return "Incomplete (Major Gaps)"
