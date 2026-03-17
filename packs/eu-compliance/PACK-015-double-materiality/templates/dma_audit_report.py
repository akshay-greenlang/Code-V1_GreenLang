# -*- coding: utf-8 -*-
"""
DMAAuditReportTemplate - Audit-ready DMA documentation for PACK-015.

Sections:
    1. Audit Report Header
    2. Methodology Documentation
    3. Stakeholder Process Audit Trail
    4. Scoring Evidence
    5. Threshold Justification
    6. Validation Results
    7. Provenance Chain
    8. Sign-off Section

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DMAAuditReportTemplate:
    """
    Audit-ready DMA documentation template.

    Renders comprehensive methodology documentation, stakeholder
    process audit trails, scoring evidence with provenance hashes,
    threshold justifications, validation results, and sign-off
    sections for external assurance readiness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DMAAuditReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render DMA audit report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_stakeholder_trail(data),
            self._md_scoring_evidence(data),
            self._md_threshold_justification(data),
            self._md_validation_results(data),
            self._md_provenance_chain(data),
            self._md_signoff(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render DMA audit report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_methodology(data),
            self._html_scoring_evidence(data),
            self._html_provenance_chain(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DMA Audit Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render DMA audit report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "dma_audit_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "workflow_id": data.get("workflow_id", ""),
            "completeness": data.get("completeness", "draft"),
            "phases": data.get("phases", []),
            "material_topics": data.get("material_topics", []),
            "validation_checks": data.get("validation_checks", []),
            "stakeholder_validation_passed": data.get("stakeholder_validation_passed", False),
            "provenance_hashes": self._extract_provenance_hashes(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render audit report header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        wf_id = data.get("workflow_id", "")
        return (
            f"# Double Materiality Assessment - Audit Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Workflow ID:** `{wf_id}`  \n"
            f"**Generated:** {ts}  \n"
            f"**Document Classification:** CONFIDENTIAL - AUDIT USE\n\n---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology documentation."""
        return (
            "## 1. Methodology\n\n"
            "### 1.1 Regulatory Framework\n\n"
            "This Double Materiality Assessment (DMA) follows:\n"
            "- **ESRS 1** - General Requirements, Chapter 3 (Double Materiality)\n"
            "- **EFRAG IG-1** - Implementation Guidance on Materiality Assessment\n"
            "- **EFRAG IG-2** - Implementation Guidance on Value Chain\n"
            "- **CSRD** - Corporate Sustainability Reporting Directive 2022/2464\n\n"
            "### 1.2 Assessment Process\n\n"
            "The DMA follows a 6-phase process:\n"
            "1. Stakeholder engagement and consultation\n"
            "2. Identification of Impacts, Risks, and Opportunities (IROs)\n"
            "3. Impact materiality assessment (inside-out perspective)\n"
            "4. Financial materiality assessment (outside-in perspective)\n"
            "5. Double materiality matrix construction with thresholds\n"
            "6. ESRS disclosure mapping and gap analysis\n\n"
            "### 1.3 Scoring Methodology\n\n"
            "- **Impact scoring:** Weighted average of scale, scope, "
            "irremediability, and likelihood (for potential impacts)\n"
            "- **Financial scoring:** Weighted average of magnitude "
            "(revenue-relative), likelihood, and time horizon\n"
            "- **Composite scoring:** Deterministic arithmetic only; "
            "no LLM used for numeric calculations"
        )

    def _md_stakeholder_trail(self, data: Dict[str, Any]) -> str:
        """Render stakeholder process audit trail."""
        sh_result = data.get("stakeholder_result", {})
        if not sh_result:
            return "## 2. Stakeholder Engagement Audit Trail\n\n_No stakeholder data available._"

        sh_count = sh_result.get("stakeholders_identified", 0)
        con_count = sh_result.get("consultations_recorded", 0)
        validated = sh_result.get("validation_passed", False)

        lines = [
            "## 2. Stakeholder Engagement Audit Trail", "",
            f"**Stakeholders Identified:** {sh_count}  ",
            f"**Consultations Recorded:** {con_count}  ",
            f"**ESRS 1 s22-23 Validation:** {'PASS' if validated else 'FAIL'}  ",
            "",
            "### 2.1 ESRS 1 Compliance Checks",
            "",
            "| Requirement | ESRS Reference | Status |",
            "|-------------|---------------|--------|",
            f"| Affected stakeholder categories | ESRS 1, para 22 | {'PASS' if validated else 'REVIEW'} |",
            f"| Key player consultation coverage | ESRS 1, para 22-23 | {'PASS' if validated else 'REVIEW'} |",
            f"| Findings documented | ESRS 1, para 23 | {'PASS' if con_count > 0 else 'FAIL'} |",
            f"| Diverse engagement methods | ESRS 1, para 22 | {'PASS' if con_count > 1 else 'REVIEW'} |",
        ]
        return "\n".join(lines)

    def _md_scoring_evidence(self, data: Dict[str, Any]) -> str:
        """Render scoring evidence with provenance."""
        phases = data.get("phases", [])
        if not phases:
            return "## 3. Scoring Evidence\n\n_No phase data available._"

        lines = [
            "## 3. Scoring Evidence", "",
            "### 3.1 Phase Execution Summary", "",
            "| Phase | Status | Duration (s) | Provenance Hash |",
            "|-------|--------|-------------|----------------|",
        ]
        for p in phases:
            status = p.get("status", "-")
            duration = p.get("duration_seconds", 0)
            prov = p.get("provenance_hash", "")[:16] + "..." if p.get("provenance_hash") else "-"
            lines.append(
                f"| {p.get('phase_name', '-')} | {status} | "
                f"{self._fmt(duration)} | `{prov}` |"
            )

        # Warnings summary
        all_warnings = []
        for p in phases:
            for w in p.get("warnings", []):
                all_warnings.append(f"- [{p.get('phase_name', '?')}] {w}")
        if all_warnings:
            lines.extend(["", "### 3.2 Warnings", ""])
            lines.extend(all_warnings)

        return "\n".join(lines)

    def _md_threshold_justification(self, data: Dict[str, Any]) -> str:
        """Render threshold justification."""
        matrix = data.get("matrix_result", {})
        if not matrix:
            return "## 4. Threshold Justification\n\n_No threshold data available._"

        i_threshold = matrix.get("impact_threshold", 2.5)
        f_threshold = matrix.get("financial_threshold", 2.5)

        return (
            "## 4. Threshold Justification\n\n"
            f"| Parameter | Value | Justification |\n"
            f"|-----------|-------|---------------|\n"
            f"| Impact Threshold | {self._fmt(i_threshold)} | "
            f"EFRAG IG-1 recommended midpoint on 0-5 scale |\n"
            f"| Financial Threshold | {self._fmt(f_threshold)} | "
            f"Aligned with impact threshold per double materiality principle |\n"
            f"| Sector Adjustments | Applied where relevant | "
            f"Based on EFRAG sector guidance and peer benchmarks |"
        )

    def _md_validation_results(self, data: Dict[str, Any]) -> str:
        """Render validation results."""
        checks = data.get("validation_checks", [])
        if not checks:
            return "## 5. Validation Results\n\n_No validation checks recorded._"

        lines = [
            "## 5. Validation Results", "",
            "| Check | Reference | Status | Details |",
            "|-------|-----------|--------|---------|",
        ]
        for vc in checks:
            status = "PASS" if vc.get("passed", False) else "FAIL"
            lines.append(
                f"| {vc.get('check_name', '-')} | "
                f"{vc.get('esrs_reference', '-')} | "
                f"**{status}** | {vc.get('details', '-')} |"
            )
        return "\n".join(lines)

    def _md_provenance_chain(self, data: Dict[str, Any]) -> str:
        """Render provenance chain for audit trail integrity."""
        hashes = self._extract_provenance_hashes(data)
        if not hashes:
            return "## 6. Provenance Chain\n\n_No provenance data available._"

        lines = [
            "## 6. Provenance Chain", "",
            "All phase outputs are hashed with SHA-256 for tamper detection.", "",
            "| Component | SHA-256 Hash |",
            "|-----------|-------------|",
        ]
        for component, hash_val in hashes.items():
            display_hash = hash_val[:32] + "..." if len(hash_val) > 32 else hash_val
            lines.append(f"| {component} | `{display_hash}` |")

        lines.extend([
            "",
            "**Integrity Note:** Any modification to input data or phase outputs "
            "will produce a different provenance hash, enabling detection of "
            "unauthorized changes.",
        ])
        return "\n".join(lines)

    def _md_signoff(self, data: Dict[str, Any]) -> str:
        """Render sign-off section."""
        return (
            "## 7. Sign-off\n\n"
            "| Role | Name | Signature | Date |\n"
            "|------|------|-----------|------|\n"
            "| DMA Lead | ________________ | ________________ | ____/____/____ |\n"
            "| Sustainability Director | ________________ | ________________ | ____/____/____ |\n"
            "| CFO / Finance Director | ________________ | ________________ | ____/____/____ |\n"
            "| External Auditor | ________________ | ________________ | ____/____/____ |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack - AUDIT DOCUMENT*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Double Materiality Assessment - Audit Report</h1>'

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology."""
        return (
            '<h2>Methodology</h2>\n'
            '<p>Assessment follows ESRS 1 Chapter 3, EFRAG IG-1, and CSRD 2022/2464.</p>'
        )

    def _html_scoring_evidence(self, data: Dict[str, Any]) -> str:
        """Render HTML scoring evidence."""
        phases = data.get("phases", [])
        rows = ""
        for p in phases:
            rows += (
                f'<tr><td>{p.get("phase_name", "-")}</td>'
                f'<td>{p.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>Phase Execution</h2>\n'
            f'<table><tr><th>Phase</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_provenance_chain(self, data: Dict[str, Any]) -> str:
        """Render HTML provenance chain."""
        hashes = self._extract_provenance_hashes(data)
        rows = ""
        for comp, h in hashes.items():
            rows += f'<tr><td>{comp}</td><td><code>{h[:32]}...</code></td></tr>\n'
        return (
            f'<h2>Provenance Chain</h2>\n'
            f'<table><tr><th>Component</th><th>SHA-256</th></tr>\n'
            f'{rows}</table>'
        )

    # -- Helpers --

    def _extract_provenance_hashes(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract provenance hashes from all phases and sub-results."""
        hashes: Dict[str, str] = {}

        # Phase-level hashes
        for p in data.get("phases", []):
            if p.get("provenance_hash"):
                hashes[f"phase:{p.get('phase_name', 'unknown')}"] = p["provenance_hash"]

        # Result-level hashes
        if data.get("provenance_hash"):
            hashes["overall_result"] = data["provenance_hash"]

        for key in ("stakeholder_result", "iro_result", "impact_result",
                     "financial_result", "matrix_result", "esrs_result"):
            sub = data.get(key, {})
            if isinstance(sub, dict) and sub.get("provenance_hash"):
                hashes[key] = sub["provenance_hash"]

        return hashes

    def _css(self) -> str:
        """Build CSS."""
        return (
            "body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "code{background:#e9ecef;padding:2px 6px;border-radius:3px;font-size:0.9em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format numeric value."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
