"""
MultiRegulationAuditTrailTemplate - Consolidated provenance and evidence mapping.

This module implements the MultiRegulationAuditTrailTemplate for PACK-009
EU Climate Compliance Bundle. It renders consolidated provenance reports
with cross-regulation evidence mapping, evidence reuse matrices, per-
regulation evidence completeness, SHA-256 hashes for all evidence items,
and audit readiness scores per regulation.

Example:
    >>> template = MultiRegulationAuditTrailTemplate()
    >>> data = AuditTrailData(
    ...     evidence_items=[...],
    ...     mappings=[...],
    ...     completeness={"CSRD": 95.0, ...},
    ...     hashes={"doc1": "abc123..."},
    ... )
    >>> html = template.render(data, fmt="html")
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic data models
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A single piece of audit evidence."""

    evidence_id: str = Field(..., description="Unique evidence identifier")
    title: str = Field(..., description="Evidence title/document name")
    description: str = Field("", description="Evidence description")
    document_type: str = Field("", description="Type: invoice, report, certificate, calculation, etc.")
    source_system: str = Field("", description="Source system or process")
    file_reference: str = Field("", description="File path or reference ID")
    sha256_hash: str = Field("", description="SHA-256 hash of the evidence content")
    regulations_served: List[str] = Field(
        default_factory=list, description="Regulations this evidence supports"
    )
    requirements_served: List[str] = Field(
        default_factory=list, description="Specific requirement IDs satisfied"
    )
    date_captured: str = Field("", description="ISO date when evidence was captured")
    date_verified: str = Field("", description="ISO date when evidence was verified")
    verified_by: str = Field("", description="Verifier identity")
    status: str = Field("active", description="active, expired, pending_review, rejected")
    retention_until: str = Field("", description="Retention date ISO string")
    notes: str = Field("", description="Additional notes")


class EvidenceMapping(BaseModel):
    """Mapping of evidence to regulation requirements."""

    mapping_id: str = Field(..., description="Mapping identifier")
    evidence_id: str = Field(..., description="Evidence item ID")
    regulation: str = Field(..., description="Regulation code")
    requirement_id: str = Field(..., description="Regulation requirement ID")
    requirement_description: str = Field("", description="Requirement description")
    coverage: str = Field("full", description="full, partial, supplementary")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Mapping confidence 0-1")
    notes: str = Field("", description="Mapping notes")


class RegulationReadiness(BaseModel):
    """Audit readiness assessment for one regulation."""

    regulation: str = Field(..., description="Regulation code")
    readiness_score: float = Field(0.0, ge=0.0, le=100.0, description="Readiness score 0-100")
    total_requirements: int = Field(0, ge=0, description="Total requirements")
    requirements_covered: int = Field(0, ge=0, description="Requirements with evidence")
    requirements_partial: int = Field(0, ge=0, description="Requirements with partial evidence")
    requirements_missing: int = Field(0, ge=0, description="Requirements without evidence")
    evidence_count: int = Field(0, ge=0, description="Number of evidence items")
    reused_evidence_count: int = Field(0, ge=0, description="Evidence shared with other regs")


class AuditTrailConfig(BaseModel):
    """Configuration for the audit trail template."""

    title: str = Field(
        "Multi-Regulation Audit Trail Report",
        description="Report title",
    )
    show_hashes: bool = Field(True, description="Whether to display SHA-256 hashes")
    show_expired: bool = Field(False, description="Whether to show expired evidence")
    regulations: List[str] = Field(
        default_factory=lambda: ["CSRD", "CBAM", "EU_TAXONOMY", "SFDR"],
        description="Regulation codes",
    )
    readiness_threshold_green: float = Field(90.0, description="Green readiness threshold")
    readiness_threshold_amber: float = Field(70.0, description="Amber readiness threshold")


class AuditTrailData(BaseModel):
    """Input data for the multi-regulation audit trail report."""

    evidence_items: List[EvidenceItem] = Field(
        default_factory=list, description="All evidence items"
    )
    mappings: List[EvidenceMapping] = Field(
        default_factory=list, description="Evidence-to-requirement mappings"
    )
    completeness: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation evidence completeness %"
    )
    hashes: Dict[str, str] = Field(
        default_factory=dict, description="Map of evidence_id to SHA-256 hash"
    )
    regulation_readiness: List[RegulationReadiness] = Field(
        default_factory=list, description="Per-regulation readiness assessments"
    )
    total_evidence_items: int = Field(0, ge=0, description="Total evidence items")
    total_requirements_covered: int = Field(0, ge=0, description="Total requirements covered")
    total_requirements: int = Field(0, ge=0, description="Total requirements across regulations")
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")

    @field_validator("evidence_items")
    @classmethod
    def validate_evidence_present(cls, v: List[EvidenceItem]) -> List[EvidenceItem]:
        """Ensure at least one evidence item is provided."""
        if not v:
            raise ValueError("evidence_items must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class MultiRegulationAuditTrailTemplate:
    """
    Multi-regulation audit trail template for consolidated provenance reporting.

    Generates audit trail reports with evidence inventories, cross-regulation
    reuse matrices, per-regulation completeness and readiness scores,
    SHA-256 hashes for all evidence, and overall audit readiness assessment.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    STATUS_COLORS = {
        "active": {"hex": "#2ecc71", "label": "Active"},
        "pending_review": {"hex": "#f39c12", "label": "Pending Review"},
        "expired": {"hex": "#e74c3c", "label": "Expired"},
        "rejected": {"hex": "#c0392b", "label": "Rejected"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize MultiRegulationAuditTrailTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = AuditTrailConfig(**raw) if raw else AuditTrailConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: AuditTrailData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the audit trail report in the specified format.

        Args:
            data: Validated AuditTrailData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output.

        Raises:
            ValueError: If fmt is unsupported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    def render_markdown(self, data: AuditTrailData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_readiness_scores(data),
            self._md_evidence_reuse_matrix(data),
            self._md_evidence_inventory(data),
            self._md_requirement_mappings(data),
            self._md_hash_registry(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: AuditTrailData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_readiness_scores(data),
            self._html_evidence_reuse_matrix(data),
            self._html_evidence_inventory(data),
            self._html_requirement_mappings(data),
            self._html_hash_registry(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: AuditTrailData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "multi_regulation_audit_trail",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "summary": self._json_summary(data),
            "readiness_scores": self._json_readiness(data),
            "evidence_reuse": self._json_evidence_reuse(data),
            "evidence_inventory": self._json_evidence_inventory(data),
            "requirement_mappings": self._json_mappings(data),
            "hash_registry": self._json_hashes(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _filtered_evidence(self, data: AuditTrailData) -> List[EvidenceItem]:
        """Return evidence items filtered by config settings."""
        items = data.evidence_items
        if not self.config.show_expired:
            items = [e for e in items if e.status != "expired"]
        return sorted(items, key=lambda e: e.evidence_id)

    def _build_evidence_lookup(self, data: AuditTrailData) -> Dict[str, EvidenceItem]:
        """Build lookup dict for evidence items by ID."""
        return {e.evidence_id: e for e in data.evidence_items}

    def _evidence_reuse_counts(self, data: AuditTrailData) -> Dict[str, int]:
        """Count how many regulations each evidence item serves."""
        counts: Dict[str, int] = {}
        for item in data.evidence_items:
            counts[item.evidence_id] = len(item.regulations_served)
        return counts

    def _reuse_matrix_data(self, data: AuditTrailData) -> Dict[str, Dict[str, int]]:
        """Build regulation-pair evidence sharing matrix."""
        regs = self.config.regulations
        matrix: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in regs} for r in regs}
        for item in data.evidence_items:
            served = [r for r in item.regulations_served if r in regs]
            for i, reg_a in enumerate(served):
                for reg_b in served[i + 1:]:
                    matrix[reg_a][reg_b] += 1
                    matrix[reg_b][reg_a] += 1
        return matrix

    def _readiness_color(self, score: float) -> str:
        """Return hex color based on readiness score."""
        if score >= self.config.readiness_threshold_green:
            return "#2ecc71"
        elif score >= self.config.readiness_threshold_amber:
            return "#f39c12"
        return "#e74c3c"

    def _readiness_label(self, score: float) -> str:
        """Return text label for readiness score."""
        if score >= self.config.readiness_threshold_green:
            return "READY"
        elif score >= self.config.readiness_threshold_amber:
            return "PARTIAL"
        return "NOT READY"

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: AuditTrailData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_summary(self, data: AuditTrailData) -> str:
        """Build summary section."""
        total_items = data.total_evidence_items or len(data.evidence_items)
        reuse_counts = self._evidence_reuse_counts(data)
        multi_reg = sum(1 for c in reuse_counts.values() if c >= 2)
        active = sum(1 for e in data.evidence_items if e.status == "active")
        pending = sum(1 for e in data.evidence_items if e.status == "pending_review")
        return (
            "## Summary\n\n"
            f"- **Total Evidence Items:** {total_items}\n"
            f"- **Active:** {active} | **Pending Review:** {pending}\n"
            f"- **Multi-Regulation Evidence:** {multi_reg} "
            f"(reused across 2+ regulations)\n"
            f"- **Total Requirements:** {data.total_requirements}\n"
            f"- **Requirements Covered:** {data.total_requirements_covered}\n"
            f"- **Regulations:** {', '.join(self.config.regulations)}"
        )

    def _md_readiness_scores(self, data: AuditTrailData) -> str:
        """Build per-regulation readiness scores."""
        if not data.regulation_readiness:
            if data.completeness:
                header = (
                    "## Audit Readiness by Regulation\n\n"
                    "| Regulation | Completeness | Status |\n"
                    "|------------|-------------|--------|\n"
                )
                rows: List[str] = []
                for reg, pct in sorted(data.completeness.items()):
                    label = self._readiness_label(pct)
                    rows.append(f"| {reg} | {pct:.1f}% | {label} |")
                return header + "\n".join(rows)
            return ""
        header = (
            "## Audit Readiness by Regulation\n\n"
            "| Regulation | Score | Total Reqs | Covered | Partial | Missing | Evidence | Reused | Status |\n"
            "|------------|-------|-----------|---------|---------|---------|----------|--------|--------|\n"
        )
        rows = []
        for rr in sorted(data.regulation_readiness, key=lambda r: r.regulation):
            label = self._readiness_label(rr.readiness_score)
            rows.append(
                f"| {rr.regulation} | {rr.readiness_score:.1f} | "
                f"{rr.total_requirements} | {rr.requirements_covered} | "
                f"{rr.requirements_partial} | {rr.requirements_missing} | "
                f"{rr.evidence_count} | {rr.reused_evidence_count} | {label} |"
            )
        return header + "\n".join(rows)

    def _md_evidence_reuse_matrix(self, data: AuditTrailData) -> str:
        """Build evidence reuse matrix."""
        regs = self.config.regulations
        matrix = self._reuse_matrix_data(data)
        header = "## Evidence Reuse Matrix\n\n"
        header += "| | " + " | ".join(regs) + " |\n"
        header += "|---" + "|---" * len(regs) + "|\n"
        rows: List[str] = []
        for reg_a in regs:
            cells: List[str] = [f"**{reg_a}**"]
            for reg_b in regs:
                if reg_a == reg_b:
                    cells.append("---")
                else:
                    count = matrix.get(reg_a, {}).get(reg_b, 0)
                    cells.append(str(count))
            rows.append("| " + " | ".join(cells) + " |")
        return header + "\n".join(rows)

    def _md_evidence_inventory(self, data: AuditTrailData) -> str:
        """Build evidence inventory table."""
        items = self._filtered_evidence(data)
        if not items:
            return "## Evidence Inventory\n\n*No evidence items to display.*"
        header = (
            "## Evidence Inventory\n\n"
            "| ID | Title | Type | Regulations | Status | Captured | Hash |\n"
            "|----|-------|------|-------------|--------|----------|------|\n"
        )
        rows: List[str] = []
        for item in items:
            regs = ", ".join(item.regulations_served) if item.regulations_served else "N/A"
            hash_display = item.sha256_hash[:16] + "..." if item.sha256_hash else "N/A"
            if not self.config.show_hashes:
                hash_display = "---"
            rows.append(
                f"| {item.evidence_id} | {item.title} | "
                f"{item.document_type or 'N/A'} | {regs} | "
                f"{item.status.replace('_', ' ').title()} | "
                f"{item.date_captured[:10] if item.date_captured else 'N/A'} | "
                f"`{hash_display}` |"
            )
        return header + "\n".join(rows)

    def _md_requirement_mappings(self, data: AuditTrailData) -> str:
        """Build requirement mappings table."""
        if not data.mappings:
            return ""
        header = (
            "## Requirement Mappings\n\n"
            "| Evidence | Regulation | Requirement | Coverage | Confidence |\n"
            "|----------|------------|-------------|----------|------------|\n"
        )
        rows: List[str] = []
        for m in sorted(data.mappings, key=lambda x: (x.regulation, x.requirement_id)):
            conf_pct = m.confidence * 100
            rows.append(
                f"| {m.evidence_id} | {m.regulation} | "
                f"{m.requirement_id} | {m.coverage.title()} | {conf_pct:.0f}% |"
            )
        return header + "\n".join(rows)

    def _md_hash_registry(self, data: AuditTrailData) -> str:
        """Build hash registry section."""
        if not self.config.show_hashes:
            return ""
        all_hashes: Dict[str, str] = dict(data.hashes)
        for item in data.evidence_items:
            if item.sha256_hash and item.evidence_id not in all_hashes:
                all_hashes[item.evidence_id] = item.sha256_hash
        if not all_hashes:
            return ""
        header = (
            "## SHA-256 Hash Registry\n\n"
            "| Evidence ID | SHA-256 Hash |\n"
            "|-------------|-------------|\n"
        )
        rows: List[str] = []
        for eid, h in sorted(all_hashes.items()):
            rows.append(f"| {eid} | `{h}` |")
        return header + "\n".join(rows)

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            "*This audit trail report provides a consolidated view of evidence "
            "across all bundled regulations. All hashes are SHA-256.*\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: MultiRegulationAuditTrailTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: AuditTrailData) -> str:
        """Build HTML header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            '<div class="report-header">'
            f'<h1>{self.config.title}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item">Organization: {org}</div>'
            f'<div class="meta-item">Period: {period}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_summary(self, data: AuditTrailData) -> str:
        """Build HTML summary cards."""
        total_items = data.total_evidence_items or len(data.evidence_items)
        reuse_counts = self._evidence_reuse_counts(data)
        multi_reg = sum(1 for c in reuse_counts.values() if c >= 2)
        active = sum(1 for e in data.evidence_items if e.status == "active")
        cards = (
            f'<div class="stat-card"><span class="stat-val">{total_items}</span>'
            f'<span class="stat-lbl">Total Evidence</span></div>'
            f'<div class="stat-card"><span class="stat-val">{active}</span>'
            f'<span class="stat-lbl">Active</span></div>'
            f'<div class="stat-card"><span class="stat-val">{multi_reg}</span>'
            f'<span class="stat-lbl">Multi-Reg Evidence</span></div>'
            f'<div class="stat-card"><span class="stat-val">{data.total_requirements}</span>'
            f'<span class="stat-lbl">Total Requirements</span></div>'
            f'<div class="stat-card"><span class="stat-val">{data.total_requirements_covered}</span>'
            f'<span class="stat-lbl">Covered</span></div>'
        )
        return f'<div class="section"><h2>Summary</h2><div class="stat-grid">{cards}</div></div>'

    def _html_readiness_scores(self, data: AuditTrailData) -> str:
        """Build HTML readiness scores."""
        if not data.regulation_readiness and not data.completeness:
            return ""
        if not data.regulation_readiness and data.completeness:
            rows = ""
            for reg, pct in sorted(data.completeness.items()):
                color = self._readiness_color(pct)
                label = self._readiness_label(pct)
                rows += (
                    f'<tr><td><strong>{reg}</strong></td>'
                    f'<td><div class="progress-bar"><div class="progress-fill" '
                    f'style="width:{pct:.0f}%;background:{color}"></div></div>'
                    f'{pct:.1f}%</td>'
                    f'<td><span class="status-badge" style="background:{color}">'
                    f'{label}</span></td></tr>'
                )
            return (
                '<div class="section"><h2>Audit Readiness by Regulation</h2>'
                '<table><thead><tr>'
                '<th>Regulation</th><th>Completeness</th><th>Status</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>'
            )
        rows = ""
        for rr in sorted(data.regulation_readiness, key=lambda r: r.regulation):
            color = self._readiness_color(rr.readiness_score)
            label = self._readiness_label(rr.readiness_score)
            rows += (
                f'<tr><td><strong>{rr.regulation}</strong></td>'
                f'<td class="num" style="color:{color};font-weight:bold">'
                f'{rr.readiness_score:.1f}</td>'
                f'<td class="num">{rr.total_requirements}</td>'
                f'<td class="num">{rr.requirements_covered}</td>'
                f'<td class="num">{rr.requirements_partial}</td>'
                f'<td class="num">{rr.requirements_missing}</td>'
                f'<td class="num">{rr.evidence_count}</td>'
                f'<td class="num">{rr.reused_evidence_count}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{label}</span></td></tr>'
            )
        return (
            '<div class="section"><h2>Audit Readiness by Regulation</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Score</th><th>Total Reqs</th>'
            '<th>Covered</th><th>Partial</th><th>Missing</th>'
            '<th>Evidence</th><th>Reused</th><th>Status</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_evidence_reuse_matrix(self, data: AuditTrailData) -> str:
        """Build HTML evidence reuse matrix."""
        regs = self.config.regulations
        matrix = self._reuse_matrix_data(data)
        header_cells = "".join(f"<th>{r}</th>" for r in regs)
        rows = ""
        for reg_a in regs:
            cells = f"<td><strong>{reg_a}</strong></td>"
            for reg_b in regs:
                if reg_a == reg_b:
                    cells += '<td class="matrix-diag">---</td>'
                else:
                    count = matrix.get(reg_a, {}).get(reg_b, 0)
                    bg = "#e8f8f5" if count > 0 else "#fff"
                    cells += f'<td class="num" style="background:{bg}">{count}</td>'
            rows += f"<tr>{cells}</tr>"
        return (
            '<div class="section"><h2>Evidence Reuse Matrix</h2>'
            '<p>Number of evidence items shared between each regulation pair.</p>'
            f'<table><thead><tr><th></th>{header_cells}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    def _html_evidence_inventory(self, data: AuditTrailData) -> str:
        """Build HTML evidence inventory table."""
        items = self._filtered_evidence(data)
        if not items:
            return (
                '<div class="section"><h2>Evidence Inventory</h2>'
                '<p class="note">No evidence items to display.</p></div>'
            )
        rows = ""
        for item in items:
            regs = ", ".join(item.regulations_served) if item.regulations_served else "N/A"
            status_info = self.STATUS_COLORS.get(item.status, {"hex": "#95a5a6", "label": item.status})
            hash_display = item.sha256_hash[:16] + "..." if item.sha256_hash else "N/A"
            if not self.config.show_hashes:
                hash_display = "---"
            rows += (
                f'<tr><td>{item.evidence_id}</td>'
                f'<td>{item.title}</td>'
                f'<td>{item.document_type or "N/A"}</td>'
                f'<td>{regs}</td>'
                f'<td><span class="status-badge" style="background:{status_info["hex"]}">'
                f'{status_info["label"]}</span></td>'
                f'<td>{item.date_captured[:10] if item.date_captured else "N/A"}</td>'
                f'<td><code>{hash_display}</code></td></tr>'
            )
        return (
            '<div class="section"><h2>Evidence Inventory</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Title</th><th>Type</th>'
            '<th>Regulations</th><th>Status</th><th>Captured</th><th>Hash</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_requirement_mappings(self, data: AuditTrailData) -> str:
        """Build HTML requirement mappings table."""
        if not data.mappings:
            return ""
        rows = ""
        for m in sorted(data.mappings, key=lambda x: (x.regulation, x.requirement_id)):
            conf_pct = m.confidence * 100
            conf_color = "#2ecc71" if conf_pct >= 90 else "#f39c12" if conf_pct >= 70 else "#e74c3c"
            cov_color = "#2ecc71" if m.coverage == "full" else "#f39c12" if m.coverage == "partial" else "#3498db"
            rows += (
                f'<tr><td>{m.evidence_id}</td>'
                f'<td>{m.regulation}</td>'
                f'<td>{m.requirement_id}</td>'
                f'<td>{m.requirement_description}</td>'
                f'<td><span class="cov-badge" style="background:{cov_color}">'
                f'{m.coverage.title()}</span></td>'
                f'<td class="num" style="color:{conf_color}">{conf_pct:.0f}%</td></tr>'
            )
        return (
            '<div class="section"><h2>Requirement Mappings</h2>'
            '<table><thead><tr>'
            '<th>Evidence</th><th>Regulation</th><th>Requirement</th>'
            '<th>Description</th><th>Coverage</th><th>Confidence</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_hash_registry(self, data: AuditTrailData) -> str:
        """Build HTML hash registry."""
        if not self.config.show_hashes:
            return ""
        all_hashes: Dict[str, str] = dict(data.hashes)
        for item in data.evidence_items:
            if item.sha256_hash and item.evidence_id not in all_hashes:
                all_hashes[item.evidence_id] = item.sha256_hash
        if not all_hashes:
            return ""
        rows = ""
        for eid, h in sorted(all_hashes.items()):
            rows += f'<tr><td>{eid}</td><td><code>{h}</code></td></tr>'
        return (
            '<div class="section"><h2>SHA-256 Hash Registry</h2>'
            '<table><thead><tr><th>Evidence ID</th><th>SHA-256 Hash</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_summary(self, data: AuditTrailData) -> Dict[str, Any]:
        """Build JSON summary."""
        reuse_counts = self._evidence_reuse_counts(data)
        multi_reg = sum(1 for c in reuse_counts.values() if c >= 2)
        return {
            "total_evidence_items": data.total_evidence_items or len(data.evidence_items),
            "active_items": sum(1 for e in data.evidence_items if e.status == "active"),
            "multi_regulation_evidence": multi_reg,
            "total_requirements": data.total_requirements,
            "total_requirements_covered": data.total_requirements_covered,
            "regulations": self.config.regulations,
        }

    def _json_readiness(self, data: AuditTrailData) -> List[Dict[str, Any]]:
        """Build JSON readiness scores."""
        if data.regulation_readiness:
            return [
                {
                    "regulation": rr.regulation,
                    "readiness_score": round(rr.readiness_score, 1),
                    "status": self._readiness_label(rr.readiness_score),
                    "total_requirements": rr.total_requirements,
                    "requirements_covered": rr.requirements_covered,
                    "requirements_partial": rr.requirements_partial,
                    "requirements_missing": rr.requirements_missing,
                    "evidence_count": rr.evidence_count,
                    "reused_evidence_count": rr.reused_evidence_count,
                }
                for rr in sorted(data.regulation_readiness, key=lambda r: r.regulation)
            ]
        return [
            {
                "regulation": reg,
                "completeness_pct": round(pct, 1),
                "status": self._readiness_label(pct),
            }
            for reg, pct in sorted(data.completeness.items())
        ]

    def _json_evidence_reuse(self, data: AuditTrailData) -> Dict[str, Any]:
        """Build JSON evidence reuse matrix."""
        matrix = self._reuse_matrix_data(data)
        return {
            "regulations": self.config.regulations,
            "matrix": {
                reg_a: {reg_b: matrix[reg_a][reg_b] for reg_b in self.config.regulations if reg_a != reg_b}
                for reg_a in self.config.regulations
            },
        }

    def _json_evidence_inventory(self, data: AuditTrailData) -> List[Dict[str, Any]]:
        """Build JSON evidence inventory."""
        items = self._filtered_evidence(data)
        return [
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "document_type": item.document_type,
                "source_system": item.source_system,
                "regulations_served": item.regulations_served,
                "requirements_served": item.requirements_served,
                "status": item.status,
                "sha256_hash": item.sha256_hash,
                "date_captured": item.date_captured,
                "date_verified": item.date_verified,
                "verified_by": item.verified_by,
                "retention_until": item.retention_until,
            }
            for item in items
        ]

    def _json_mappings(self, data: AuditTrailData) -> List[Dict[str, Any]]:
        """Build JSON mappings."""
        return [
            {
                "mapping_id": m.mapping_id,
                "evidence_id": m.evidence_id,
                "regulation": m.regulation,
                "requirement_id": m.requirement_id,
                "requirement_description": m.requirement_description,
                "coverage": m.coverage,
                "confidence": round(m.confidence, 2),
            }
            for m in sorted(data.mappings, key=lambda x: (x.regulation, x.requirement_id))
        ]

    def _json_hashes(self, data: AuditTrailData) -> Dict[str, str]:
        """Build JSON hash registry."""
        all_hashes: Dict[str, str] = dict(data.hashes)
        for item in data.evidence_items:
            if item.sha256_hash and item.evidence_id not in all_hashes:
                all_hashes[item.evidence_id] = item.sha256_hash
        return dict(sorted(all_hashes.items()))

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:12px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".stat-grid{display:flex;flex-wrap:wrap;gap:12px}"
            ".stat-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;"
            "min-width:100px;flex:1}"
            ".stat-val{display:block;font-size:24px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d;margin-top:4px}"
            ".status-badge,.cov-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".matrix-diag{background:#ecf0f1;text-align:center;color:#95a5a6}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;display:inline-block;width:70%}"
            ".progress-fill{height:100%;border-radius:4px}"
            "code{background:#ecf0f1;padding:2px 6px;border-radius:3px;font-size:12px;"
            "word-break:break-all}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{self.config.title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: MultiRegulationAuditTrailTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
