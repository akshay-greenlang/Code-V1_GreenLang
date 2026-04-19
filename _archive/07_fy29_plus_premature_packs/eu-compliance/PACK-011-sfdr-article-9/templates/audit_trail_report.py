"""
AuditTrailReportTemplate - Data provenance and audit trail report.

This module implements the audit trail report template for PACK-011
SFDR Article 9 products. It provides comprehensive audit documentation
including provenance hashes, data source inventory, methodology
references, data quality flags, version history, and sign-off tracking.

Article 9 products are subject to heightened regulatory scrutiny and
require a complete audit trail demonstrating how sustainable investment
claims are substantiated by verifiable data and transparent methodology.

Example:
    >>> template = AuditTrailReportTemplate()
    >>> data = AuditTrailReportData(
    ...     report_info=AuditReportInfo(report_name="Q4 2025 Audit", ...),
    ...     provenance_records=[...],
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class AuditReportInfo(BaseModel):
    """Report identification for audit trail."""

    report_name: str = Field(..., min_length=1, description="Audit report name")
    fund_name: str = Field("", description="Fund name")
    isin: str = Field("", description="ISIN code")
    reporting_period_start: str = Field("", description="Period start (YYYY-MM-DD)")
    reporting_period_end: str = Field("", description="Period end (YYYY-MM-DD)")
    audit_date: str = Field("", description="Audit date (YYYY-MM-DD)")
    auditor: str = Field("", description="Auditor name or firm")
    audit_scope: str = Field(
        "", description="Scope of the audit trail"
    )
    classification: str = Field(
        "article_9", description="SFDR classification"
    )


class ProvenanceRecord(BaseModel):
    """Individual provenance hash record."""

    record_id: str = Field("", description="Unique record identifier")
    artifact_name: str = Field("", description="Name of the data artifact")
    artifact_type: str = Field(
        "",
        description="Type: report, calculation, dataset, model_output, disclosure",
    )
    sha256_hash: str = Field("", description="SHA-256 hash of the artifact")
    timestamp: str = Field("", description="Creation timestamp (ISO 8601)")
    creator: str = Field("", description="Creator agent or user")
    input_hashes: List[str] = Field(
        default_factory=list,
        description="SHA-256 hashes of input artifacts",
    )
    description: str = Field("", description="Description of the artifact")
    version: str = Field("1.0", description="Artifact version")


class DataSourceEntry(BaseModel):
    """Data source inventory entry."""

    source_id: str = Field("", description="Source identifier")
    source_name: str = Field("", description="Source name")
    provider: str = Field("", description="Data provider")
    data_type: str = Field(
        "", description="Data type: reported, estimated, modeled, proxy"
    )
    coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Data coverage (%)"
    )
    update_frequency: str = Field("", description="Update frequency")
    last_updated: str = Field("", description="Last update date")
    quality_rating: str = Field(
        "", description="Quality rating: high, medium, low"
    )
    validation_status: str = Field(
        "", description="Validation status: validated, pending, failed"
    )
    license_info: str = Field("", description="Data license information")
    contact: str = Field("", description="Data provider contact")


class MethodologyReference(BaseModel):
    """Methodology reference entry."""

    methodology_id: str = Field("", description="Methodology identifier")
    name: str = Field("", description="Methodology name")
    standard: str = Field(
        "",
        description="Standard reference (e.g., GHG Protocol, PCAF, EU Taxonomy TR)",
    )
    version: str = Field("", description="Methodology version")
    applicable_to: List[str] = Field(
        default_factory=list,
        description="Areas where this methodology applies",
    )
    key_assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions made"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Known limitations"
    )
    last_review_date: str = Field("", description="Last methodology review date")
    reviewer: str = Field("", description="Methodology reviewer")


class DataQualityFlag(BaseModel):
    """Data quality flag entry."""

    flag_id: str = Field("", description="Flag identifier")
    data_element: str = Field("", description="Affected data element")
    flag_type: str = Field(
        "",
        description="Flag type: missing, estimated, outlier, stale, inconsistent",
    )
    severity: str = Field(
        "medium", description="Severity: critical, high, medium, low, info"
    )
    description: str = Field("", description="Flag description")
    impact_assessment: str = Field(
        "", description="Impact on report accuracy"
    )
    remediation: str = Field("", description="Remediation action taken or planned")
    status: str = Field(
        "open", description="Status: open, in_progress, resolved, accepted"
    )
    raised_date: str = Field("", description="Date flag was raised")
    resolved_date: str = Field("", description="Date flag was resolved (if applicable)")


class VersionHistoryEntry(BaseModel):
    """Version history entry."""

    version: str = Field("", description="Version number")
    date: str = Field("", description="Version date")
    author: str = Field("", description="Author of changes")
    change_type: str = Field(
        "",
        description="Change type: initial, correction, update, methodology_change",
    )
    description: str = Field("", description="Description of changes")
    affected_sections: List[str] = Field(
        default_factory=list, description="Sections affected by changes"
    )
    approval_status: str = Field(
        "", description="Approval status: draft, reviewed, approved"
    )
    approver: str = Field("", description="Approver name")


class SignOffRecord(BaseModel):
    """Sign-off tracking record."""

    signoff_id: str = Field("", description="Sign-off identifier")
    role: str = Field("", description="Signer role (e.g., Head of ESG, CIO, Compliance)")
    signer_name: str = Field("", description="Signer name")
    sign_date: str = Field("", description="Sign-off date")
    status: str = Field(
        "pending",
        description="Status: pending, signed, rejected, conditional",
    )
    comments: str = Field("", description="Sign-off comments")
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions attached to sign-off (if conditional)",
    )
    digital_signature_hash: str = Field(
        "", description="Digital signature hash"
    )


class AuditTrailReportData(BaseModel):
    """Complete input data for audit trail report."""

    report_info: AuditReportInfo
    provenance_records: List[ProvenanceRecord] = Field(
        default_factory=list, description="Provenance hash records"
    )
    data_sources: List[DataSourceEntry] = Field(
        default_factory=list, description="Data source inventory"
    )
    methodology_refs: List[MethodologyReference] = Field(
        default_factory=list, description="Methodology references"
    )
    quality_flags: List[DataQualityFlag] = Field(
        default_factory=list, description="Data quality flags"
    )
    version_history: List[VersionHistoryEntry] = Field(
        default_factory=list, description="Version history"
    )
    signoffs: List[SignOffRecord] = Field(
        default_factory=list, description="Sign-off records"
    )


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AuditTrailReportTemplate:
    """
    Audit trail and data provenance report template for Article 9 products.

    Generates a comprehensive audit trail covering provenance hashes,
    data source inventory, methodology references, data quality flags,
    version history, and sign-off tracking.

    Attributes:
        config: Optional configuration dictionary.
        PACK_ID: Pack identifier (PACK-011).
        TEMPLATE_NAME: Template identifier.
        VERSION: Template version.

    Example:
        >>> template = AuditTrailReportTemplate()
        >>> md = template.render_markdown(data)
        >>> assert "Provenance" in md and "SHA-256" in md
    """

    PACK_ID = "PACK-011"
    TEMPLATE_NAME = "audit_trail_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AuditTrailReportTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    #  Public render dispatcher
    # ------------------------------------------------------------------ #

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render audit trail report in the specified format.

        Args:
            data: Report data dictionary matching AuditTrailReportData schema.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered content as string (markdown/html) or dict (json).

        Raises:
            ValueError: If format is not supported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    # ------------------------------------------------------------------ #
    #  Markdown rendering
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render audit trail report as Markdown.

        Args:
            data: Report data dictionary matching AuditTrailReportData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header(data))
        sections.append(self._md_section_1_provenance(data))
        sections.append(self._md_section_2_data_sources(data))
        sections.append(self._md_section_3_methodology(data))
        sections.append(self._md_section_4_quality_flags(data))
        sections.append(self._md_section_5_version_history(data))
        sections.append(self._md_section_6_signoffs(data))

        content = "\n\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render audit trail report as self-contained HTML.

        Args:
            data: Report data dictionary matching AuditTrailReportData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_section_1_provenance(data))
        sections.append(self._html_section_2_data_sources(data))
        sections.append(self._html_section_3_methodology(data))
        sections.append(self._html_section_4_quality_flags(data))
        sections.append(self._html_section_5_version_history(data))
        sections.append(self._html_section_6_signoffs(data))

        body = "\n".join(s for s in sections if s)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="SFDR Article 9 Audit Trail Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render audit trail report as structured JSON.

        Args:
            data: Report data dictionary matching AuditTrailReportData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "sfdr_article_9_audit_trail",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "report_info": data.get("report_info", {}),
            "provenance_records": data.get("provenance_records", []),
            "data_sources": data.get("data_sources", []),
            "methodology_refs": data.get("methodology_refs", []),
            "quality_flags": data.get("quality_flags", []),
            "version_history": data.get("version_history", []),
            "signoffs": data.get("signoffs", []),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown Section Builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown document header."""
        ri = data.get("report_info", {})
        name = ri.get("report_name", "Audit Trail Report")
        fund = ri.get("fund_name", "")
        return (
            f"# Audit Trail Report (SFDR Article 9)\n\n"
            f"**Report:** {name}\n\n"
            f"**Fund:** {fund or 'N/A'}\n\n"
            f"**ISIN:** {ri.get('isin', 'N/A') or 'N/A'}\n\n"
            f"**Period:** {ri.get('reporting_period_start', '')} to "
            f"{ri.get('reporting_period_end', '')}\n\n"
            f"**Audit Date:** {ri.get('audit_date', '')} | "
            f"**Auditor:** {ri.get('auditor', 'N/A') or 'N/A'}\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_section_1_provenance(self, data: Dict[str, Any]) -> str:
        """Section 1: Provenance hashes."""
        records = data.get("provenance_records", [])

        lines: List[str] = [
            "## 1. Provenance Hashes\n",
            f"**Total Artifacts Tracked:** {len(records)}\n",
        ]

        if records:
            lines.append(
                "| ID | Artifact | Type | SHA-256 | Timestamp | Creator |"
            )
            lines.append(
                "|----|----------|------|---------|-----------|---------|"
            )
            for r in records:
                sha = r.get("sha256_hash", "")
                sha_short = sha[:16] + "..." if len(sha) > 16 else sha
                lines.append(
                    f"| {r.get('record_id', '')} | "
                    f"{r.get('artifact_name', '')} | "
                    f"{r.get('artifact_type', '')} | "
                    f"`{sha_short}` | "
                    f"{r.get('timestamp', '')} | "
                    f"{r.get('creator', '')} |"
                )
            lines.append("")

            # Lineage details
            for r in records:
                inputs = r.get("input_hashes", [])
                if inputs:
                    lines.append(
                        f"**{r.get('record_id', '')} "
                        f"({r.get('artifact_name', '')})** depends on:\n"
                    )
                    for ih in inputs:
                        ih_short = ih[:16] + "..." if len(ih) > 16 else ih
                        lines.append(f"- `{ih_short}`")
                    lines.append("")
        else:
            lines.append("No provenance records available.")

        return "\n".join(lines)

    def _md_section_2_data_sources(self, data: Dict[str, Any]) -> str:
        """Section 2: Data source inventory."""
        sources = data.get("data_sources", [])

        lines: List[str] = [
            "## 2. Data Source Inventory\n",
            f"**Total Sources:** {len(sources)}\n",
        ]

        if sources:
            lines.append(
                "| ID | Source | Provider | Type | Coverage | Quality | Updated |"
            )
            lines.append(
                "|----|--------|----------|------|----------|---------|---------|"
            )
            for s in sources:
                lines.append(
                    f"| {s.get('source_id', '')} | "
                    f"{s.get('source_name', '')} | "
                    f"{s.get('provider', '')} | "
                    f"{s.get('data_type', '')} | "
                    f"{s.get('coverage_pct', 0.0):.0f}% | "
                    f"{s.get('quality_rating', '')} | "
                    f"{s.get('last_updated', '')} |"
                )
            lines.append("")

            # Validation status summary
            validated = sum(1 for s in sources if s.get("validation_status") == "validated")
            pending = sum(1 for s in sources if s.get("validation_status") == "pending")
            failed = sum(1 for s in sources if s.get("validation_status") == "failed")
            lines.append(
                f"**Validation Summary:** {validated} validated, "
                f"{pending} pending, {failed} failed"
            )
        else:
            lines.append("No data sources documented.")

        return "\n".join(lines)

    def _md_section_3_methodology(self, data: Dict[str, Any]) -> str:
        """Section 3: Methodology references."""
        methods = data.get("methodology_refs", [])

        lines: List[str] = [
            "## 3. Methodology References\n",
            f"**Total Methodologies:** {len(methods)}\n",
        ]

        if methods:
            for m in methods:
                lines.append(
                    f"### {m.get('methodology_id', '')} - {m.get('name', '')}\n"
                )
                lines.append("| Field | Value |")
                lines.append("|-------|-------|")
                lines.append(f"| **Standard** | {m.get('standard', 'N/A')} |")
                lines.append(f"| **Version** | {m.get('version', 'N/A')} |")
                lines.append(f"| **Last Review** | {m.get('last_review_date', 'N/A')} |")
                lines.append(f"| **Reviewer** | {m.get('reviewer', 'N/A')} |")
                lines.append("")

                applicable = m.get("applicable_to", [])
                if applicable:
                    lines.append("**Applicable To:**\n")
                    for a in applicable:
                        lines.append(f"- {a}")
                    lines.append("")

                assumptions = m.get("key_assumptions", [])
                if assumptions:
                    lines.append("**Key Assumptions:**\n")
                    for a in assumptions:
                        lines.append(f"- {a}")
                    lines.append("")

                limitations = m.get("limitations", [])
                if limitations:
                    lines.append("**Limitations:**\n")
                    for lim in limitations:
                        lines.append(f"- {lim}")
                    lines.append("")
        else:
            lines.append("No methodology references documented.")

        return "\n".join(lines)

    def _md_section_4_quality_flags(self, data: Dict[str, Any]) -> str:
        """Section 4: Data quality flags."""
        flags = data.get("quality_flags", [])

        lines: List[str] = [
            "## 4. Data Quality Flags\n",
            f"**Total Flags:** {len(flags)}\n",
        ]

        if flags:
            # Summary counts
            severity_counts: Dict[str, int] = {}
            status_counts: Dict[str, int] = {}
            for f_item in flags:
                sev = f_item.get("severity", "medium")
                stat = f_item.get("status", "open")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                status_counts[stat] = status_counts.get(stat, 0) + 1

            lines.append("### Summary\n")
            lines.append("| Severity | Count |")
            lines.append("|----------|-------|")
            for sev in ["critical", "high", "medium", "low", "info"]:
                count = severity_counts.get(sev, 0)
                if count > 0:
                    lines.append(f"| {sev.upper()} | {count} |")
            lines.append("")

            lines.append("| Status | Count |")
            lines.append("|--------|-------|")
            for stat in ["open", "in_progress", "resolved", "accepted"]:
                count = status_counts.get(stat, 0)
                if count > 0:
                    lines.append(f"| {stat.replace('_', ' ').title()} | {count} |")
            lines.append("")

            # Detail table
            lines.append("### Flag Details\n")
            lines.append(
                "| ID | Element | Type | Severity | Status | Description |"
            )
            lines.append(
                "|----|---------|------|----------|--------|-------------|"
            )
            for f_item in flags:
                lines.append(
                    f"| {f_item.get('flag_id', '')} | "
                    f"{f_item.get('data_element', '')} | "
                    f"{f_item.get('flag_type', '')} | "
                    f"{f_item.get('severity', '')} | "
                    f"{f_item.get('status', '')} | "
                    f"{f_item.get('description', '')} |"
                )
        else:
            lines.append("No data quality flags raised.")

        return "\n".join(lines)

    def _md_section_5_version_history(self, data: Dict[str, Any]) -> str:
        """Section 5: Version history."""
        versions = data.get("version_history", [])

        lines: List[str] = [
            "## 5. Version History\n",
            f"**Total Versions:** {len(versions)}\n",
        ]

        if versions:
            lines.append(
                "| Version | Date | Author | Type | Approval | Approver |"
            )
            lines.append(
                "|---------|------|--------|------|----------|----------|"
            )
            for v in versions:
                lines.append(
                    f"| {v.get('version', '')} | "
                    f"{v.get('date', '')} | "
                    f"{v.get('author', '')} | "
                    f"{v.get('change_type', '')} | "
                    f"{v.get('approval_status', '')} | "
                    f"{v.get('approver', '')} |"
                )
            lines.append("")

            # Change descriptions
            for v in versions:
                desc = v.get("description", "")
                affected = v.get("affected_sections", [])
                if desc or affected:
                    lines.append(f"**v{v.get('version', '')}:** {desc}\n")
                    if affected:
                        lines.append(
                            f"*Affected sections:* {', '.join(affected)}\n"
                        )
        else:
            lines.append("No version history available.")

        return "\n".join(lines)

    def _md_section_6_signoffs(self, data: Dict[str, Any]) -> str:
        """Section 6: Sign-off tracking."""
        signoffs = data.get("signoffs", [])

        lines: List[str] = [
            "## 6. Sign-Off Tracking\n",
        ]

        if signoffs:
            # Summary
            signed = sum(1 for s in signoffs if s.get("status") == "signed")
            pending = sum(1 for s in signoffs if s.get("status") == "pending")
            rejected = sum(1 for s in signoffs if s.get("status") == "rejected")
            conditional = sum(
                1 for s in signoffs if s.get("status") == "conditional"
            )

            lines.append(
                f"**Signed:** {signed} | **Pending:** {pending} | "
                f"**Rejected:** {rejected} | **Conditional:** {conditional}\n"
            )

            lines.append("| ID | Role | Signer | Date | Status | Comments |")
            lines.append("|-----|------|--------|------|--------|----------|")
            for s in signoffs:
                lines.append(
                    f"| {s.get('signoff_id', '')} | "
                    f"{s.get('role', '')} | "
                    f"{s.get('signer_name', '')} | "
                    f"{s.get('sign_date', '')} | "
                    f"{s.get('status', '')} | "
                    f"{s.get('comments', '')} |"
                )
            lines.append("")

            # Conditional sign-offs
            conditional_list = [
                s for s in signoffs if s.get("status") == "conditional"
            ]
            if conditional_list:
                lines.append("### Conditional Sign-Offs\n")
                for s in conditional_list:
                    conditions = s.get("conditions", [])
                    if conditions:
                        lines.append(
                            f"**{s.get('signer_name', '')} "
                            f"({s.get('role', '')}):**\n"
                        )
                        for c in conditions:
                            lines.append(f"- {c}")
                        lines.append("")
        else:
            lines.append("No sign-off records available.")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML Section Builders
    # ------------------------------------------------------------------ #

    def _html_section_1_provenance(self, data: Dict[str, Any]) -> str:
        """Build HTML provenance section."""
        records = data.get("provenance_records", [])

        parts: List[str] = [
            '<div class="section"><h2>1. Provenance Hashes</h2>',
            f"<p>Total Artifacts: {len(records)}</p>",
        ]

        if records:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>Artifact</th><th>Type</th>"
                "<th>SHA-256</th><th>Creator</th></tr>"
            )
            for r in records:
                sha = r.get("sha256_hash", "")
                sha_short = sha[:16] + "..." if len(sha) > 16 else sha
                parts.append(
                    f"<tr><td>{_esc(str(r.get('record_id', '')))}</td>"
                    f"<td>{_esc(str(r.get('artifact_name', '')))}</td>"
                    f"<td>{_esc(str(r.get('artifact_type', '')))}</td>"
                    f'<td style="font-family:monospace;font-size:0.85em;">'
                    f"{_esc(sha_short)}</td>"
                    f"<td>{_esc(str(r.get('creator', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_2_data_sources(self, data: Dict[str, Any]) -> str:
        """Build HTML data sources section."""
        sources = data.get("data_sources", [])

        parts: List[str] = [
            '<div class="section"><h2>2. Data Source Inventory</h2>',
        ]

        if sources:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>Source</th><th>Provider</th>"
                "<th>Type</th><th>Coverage</th><th>Quality</th></tr>"
            )
            for s in sources:
                quality = s.get("quality_rating", "")
                q_colors = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}
                q_color = q_colors.get(quality, "#7f8c8d")
                parts.append(
                    f"<tr><td>{_esc(str(s.get('source_id', '')))}</td>"
                    f"<td>{_esc(str(s.get('source_name', '')))}</td>"
                    f"<td>{_esc(str(s.get('provider', '')))}</td>"
                    f"<td>{_esc(str(s.get('data_type', '')))}</td>"
                    f"<td>{s.get('coverage_pct', 0.0):.0f}%</td>"
                    f'<td style="color:{q_color};font-weight:bold;">'
                    f"{_esc(quality.upper())}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_3_methodology(self, data: Dict[str, Any]) -> str:
        """Build HTML methodology section."""
        methods = data.get("methodology_refs", [])

        parts: List[str] = [
            '<div class="section"><h2>3. Methodology References</h2>',
        ]

        if methods:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>Name</th><th>Standard</th>"
                "<th>Version</th><th>Last Review</th></tr>"
            )
            for m in methods:
                parts.append(
                    f"<tr><td>{_esc(str(m.get('methodology_id', '')))}</td>"
                    f"<td>{_esc(str(m.get('name', '')))}</td>"
                    f"<td>{_esc(str(m.get('standard', '')))}</td>"
                    f"<td>{_esc(str(m.get('version', '')))}</td>"
                    f"<td>{_esc(str(m.get('last_review_date', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_4_quality_flags(self, data: Dict[str, Any]) -> str:
        """Build HTML quality flags section."""
        flags = data.get("quality_flags", [])

        parts: List[str] = [
            '<div class="section"><h2>4. Data Quality Flags</h2>',
            f"<p>Total Flags: {len(flags)}</p>",
        ]

        if flags:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>ID</th><th>Element</th><th>Type</th>"
                "<th>Severity</th><th>Status</th></tr>"
            )
            for f_item in flags:
                sev = f_item.get("severity", "medium")
                sev_colors = {
                    "critical": "#c0392b", "high": "#e74c3c",
                    "medium": "#f39c12", "low": "#27ae60", "info": "#3498db",
                }
                sev_color = sev_colors.get(sev, "#7f8c8d")
                parts.append(
                    f"<tr><td>{_esc(str(f_item.get('flag_id', '')))}</td>"
                    f"<td>{_esc(str(f_item.get('data_element', '')))}</td>"
                    f"<td>{_esc(str(f_item.get('flag_type', '')))}</td>"
                    f'<td style="color:{sev_color};font-weight:bold;">'
                    f"{_esc(sev.upper())}</td>"
                    f"<td>{_esc(str(f_item.get('status', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_5_version_history(self, data: Dict[str, Any]) -> str:
        """Build HTML version history section."""
        versions = data.get("version_history", [])

        parts: List[str] = [
            '<div class="section"><h2>5. Version History</h2>',
        ]

        if versions:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Version</th><th>Date</th><th>Author</th>"
                "<th>Type</th><th>Approval</th></tr>"
            )
            for v in versions:
                parts.append(
                    f"<tr><td>{_esc(str(v.get('version', '')))}</td>"
                    f"<td>{_esc(str(v.get('date', '')))}</td>"
                    f"<td>{_esc(str(v.get('author', '')))}</td>"
                    f"<td>{_esc(str(v.get('change_type', '')))}</td>"
                    f"<td>{_esc(str(v.get('approval_status', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    def _html_section_6_signoffs(self, data: Dict[str, Any]) -> str:
        """Build HTML sign-off section."""
        signoffs = data.get("signoffs", [])

        parts: List[str] = [
            '<div class="section"><h2>6. Sign-Off Tracking</h2>',
        ]

        if signoffs:
            parts.append('<table class="data-table">')
            parts.append(
                "<tr><th>Role</th><th>Signer</th><th>Date</th>"
                "<th>Status</th><th>Comments</th></tr>"
            )
            for s in signoffs:
                status = s.get("status", "pending")
                status_colors = {
                    "signed": "#27ae60", "pending": "#f39c12",
                    "rejected": "#e74c3c", "conditional": "#3498db",
                }
                s_color = status_colors.get(status, "#7f8c8d")
                parts.append(
                    f"<tr><td>{_esc(str(s.get('role', '')))}</td>"
                    f"<td>{_esc(str(s.get('signer_name', '')))}</td>"
                    f"<td>{_esc(str(s.get('sign_date', '')))}</td>"
                    f'<td style="color:{s_color};font-weight:bold;">'
                    f"{_esc(status.upper())}</td>"
                    f"<td>{_esc(str(s.get('comments', '')))}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</div>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Shared Utilities
    # ------------------------------------------------------------------ #

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown footer with provenance."""
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; margin: 40px auto; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #1abc9c; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; border-bottom: 1px solid #bdc3c7; "
            "padding-bottom: 5px; }\n"
            "h3 { color: #2c3e50; }\n"
            ".section { margin-bottom: 30px; padding: 15px; "
            "background: #fafafa; border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #1a5276; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; "
            "border-top: 1px solid #bdc3c7; padding-top: 10px; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID} | '
            f'{self.TEMPLATE_NAME} v{self.VERSION}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        """Compute SHA-256 provenance hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Module-level HTML escaping utility
# ---------------------------------------------------------------------------

def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
