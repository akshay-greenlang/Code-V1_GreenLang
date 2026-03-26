# -*- coding: utf-8 -*-
"""
ISAE3410EvidenceBundle - ISAE 3410 Evidence Bundle for PACK-048.

Generates an ISAE 3410-specific evidence bundle with section mapping
covering all eight core sections of the standard (engagement terms,
GHG statement, risk assessment, evidence, controls, analytical
procedures, representations, and conclusions), cross-references to
evidence items per section, and XBRL tag mapping for ISAE 3410 fields.

Regulatory References:
    - ISAE 3410: Assurance Engagements on Greenhouse Gas Statements
    - IAASB Handbook 2024: International Standards on Assurance Engagements
    - GHG Protocol Corporate Standard
    - EFRAG XBRL Taxonomy for ESRS alignment

Sections:
    1. Engagement Terms and Scope (ISAE 3410 Sec 1)
    2. GHG Statement and Methodology (Sec 2)
    3. Risk Assessment and Materiality (Sec 3)
    4. Evidence and Procedures (Sec 4)
    5. Controls Assessment (Sec 5)
    6. Analytical Procedures (Sec 6)
    7. Representations (Sec 7)
    8. Conclusions and Report (Sec 8)
    9. XBRL Tag Mapping
    10. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with evidence cross-references)
    - XBRL (tagged disclosure elements)

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    XBRL = "xbrl"


class SectionStatus(str, Enum):
    """Section completion status."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"


class EvidenceQuality(str, Enum):
    """Evidence quality rating."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class EvidenceReference(BaseModel):
    """Reference to an evidence item."""
    evidence_id: str = Field("", description="Evidence unique ID")
    title: str = Field("", description="Evidence title")
    document_type: str = Field("", description="Document type")
    quality: EvidenceQuality = Field(EvidenceQuality.MEDIUM, description="Quality")
    sha256_hash: str = Field("", description="SHA-256 hash")
    notes: str = Field("", description="Notes")


class ISAESection(BaseModel):
    """Single ISAE 3410 section."""
    section_number: int = Field(..., ge=1, le=8, description="Section number (1-8)")
    section_title: str = Field(..., description="Section title")
    description: str = Field("", description="Section description")
    isae_paragraphs: str = Field("", description="Relevant ISAE 3410 paragraphs")
    status: SectionStatus = Field(SectionStatus.NOT_STARTED, description="Status")
    completion_pct: float = Field(0.0, ge=0, le=100, description="Completion %")
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list, description="Evidence references for this section"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings for this section"
    )
    actions_required: List[str] = Field(
        default_factory=list, description="Actions still required"
    )


class XBRLMapping(BaseModel):
    """XBRL tag mapping for ISAE 3410 field."""
    field_name: str = Field(..., description="Field name")
    xbrl_tag: str = Field("", description="XBRL tag")
    taxonomy: str = Field("ESRS", description="Taxonomy (ESRS, GRI, etc.)")
    section: int = Field(0, ge=0, le=8, description="Related ISAE section")
    value: str = Field("", description="Current value")
    data_type: str = Field("string", description="Data type (string, decimal, date)")


class ISAE3410BundleInput(BaseModel):
    """Complete input model for ISAE3410EvidenceBundle."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    engagement_reference: str = Field("", description="Engagement reference")
    assurance_level: str = Field("limited", description="Assurance level")
    verifier_name: str = Field("", description="Verifier name")
    sections: List[ISAESection] = Field(
        default_factory=list, description="ISAE 3410 sections (1-8)"
    )
    xbrl_mappings: List[XBRLMapping] = Field(
        default_factory=list, description="XBRL tag mappings"
    )
    overall_completion_pct: float = Field(
        0.0, ge=0, le=100, description="Overall bundle completion %"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _status_label(status: str) -> str:
    """Return display label for status."""
    return status.replace("_", " ").title()


def _status_css(status: str) -> str:
    """Return CSS class for status."""
    mapping = {
        "complete": "sec-complete",
        "partial": "sec-partial",
        "not_started": "sec-not-started",
    }
    return mapping.get(status, "sec-partial")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ISAE3410EvidenceBundle:
    """
    ISAE 3410 evidence bundle template for PACK-048.

    Renders the complete ISAE 3410 section mapping with evidence
    cross-references, XBRL tag mapping, and multi-format export.
    All outputs include SHA-256 provenance hashing for audit-trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = ISAE3410EvidenceBundle()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> xbrl = template.render_xbrl(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ISAE3410EvidenceBundle."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        elif fmt == "xbrl":
            return self.render_xbrl(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render evidence bundle as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render evidence bundle as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render evidence bundle as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_xbrl(self, data: Dict[str, Any]) -> str:
        """Render XBRL tagged output."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_xbrl(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
        ]
        for sec in data.get("sections", []):
            sections.append(self._md_isae_section(sec))
        sections.append(self._md_xbrl_mapping(data))
        sections.append(self._md_footer(data))
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        verifier = self._get_val(data, "verifier_name", "")
        return (
            f"# ISAE 3410 Evidence Bundle - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Assurance Level:** {level.title()} | "
            f"**Verifier:** {verifier}\n\n"
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown overview table."""
        overall = data.get("overall_completion_pct", 0)
        sections = data.get("sections", [])
        lines = [
            "## Bundle Overview",
            "",
            f"**Overall Completion:** {overall:.0f}%",
            "",
            "| Section | Title | Status | Completion | Evidence Items |",
            "|---------|-------|--------|------------|----------------|",
        ]
        for sec in sections:
            ev_count = len(sec.get("evidence_refs", []))
            lines.append(
                f"| {sec.get('section_number', '')} | "
                f"{sec.get('section_title', '')} | "
                f"**{_status_label(sec.get('status', 'not_started'))}** | "
                f"{sec.get('completion_pct', 0):.0f}% | "
                f"{ev_count} |"
            )
        return "\n".join(lines)

    def _md_isae_section(self, sec: Dict[str, Any]) -> str:
        """Render Markdown for a single ISAE section."""
        num = sec.get("section_number", 0)
        title = sec.get("section_title", "")
        desc = sec.get("description", "")
        paras = sec.get("isae_paragraphs", "")
        lines = [
            f"## Section {num}: {title}",
        ]
        if paras:
            lines.append(f"*ISAE 3410 Reference: {paras}*")
        if desc:
            lines.append("")
            lines.append(desc)
        # Evidence references
        evidence = sec.get("evidence_refs", [])
        if evidence:
            lines.append("")
            lines.append("### Evidence Items")
            lines.append("")
            lines.append("| ID | Title | Type | Quality | Hash |")
            lines.append("|----|-------|------|---------|------|")
            for ev in evidence:
                eid = ev.get("evidence_id", "")[:8]
                lines.append(
                    f"| `{eid}` | "
                    f"{ev.get('title', '')} | "
                    f"{ev.get('document_type', '')} | "
                    f"{ev.get('quality', 'medium').upper()} | "
                    f"`{ev.get('sha256_hash', '')[:16]}...` |"
                )
        # Key findings
        findings = sec.get("key_findings", [])
        if findings:
            lines.append("")
            lines.append("### Key Findings")
            for f in findings:
                lines.append(f"- {f}")
        # Actions required
        actions = sec.get("actions_required", [])
        if actions:
            lines.append("")
            lines.append("### Actions Required")
            for a in actions:
                lines.append(f"- {a}")
        return "\n".join(lines)

    def _md_xbrl_mapping(self, data: Dict[str, Any]) -> str:
        """Render Markdown XBRL tag mapping."""
        mappings = data.get("xbrl_mappings", [])
        if not mappings:
            return ""
        lines = [
            "## XBRL Tag Mapping",
            "",
            "| Field | XBRL Tag | Taxonomy | Section | Value |",
            "|-------|----------|----------|---------|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('field_name', '')} | "
                f"`{m.get('xbrl_tag', '')}` | "
                f"{m.get('taxonomy', 'ESRS')} | "
                f"{m.get('section', '')} | "
                f"{m.get('value', '')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
        ]
        for sec in data.get("sections", []):
            body_parts.append(self._html_isae_section(sec))
        body_parts.append(self._html_xbrl_mapping(data))
        body_parts.append(self._html_footer(data))
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>ISAE 3410 Evidence Bundle - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".sec-complete{color:#2a9d8f;font-weight:700;}\n"
            ".sec-partial{color:#e9c46a;font-weight:700;}\n"
            ".sec-not-started{color:#e76f51;font-weight:700;}\n"
            ".hash-cell{font-family:monospace;font-size:0.75rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        verifier = self._get_val(data, "verifier_name", "")
        return (
            '<div class="section">\n'
            f"<h1>ISAE 3410 Evidence Bundle &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Level:</strong> {level.title()} | "
            f"<strong>Verifier:</strong> {verifier} | "
            f"<strong>Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        overall = data.get("overall_completion_pct", 0)
        sections = data.get("sections", [])
        rows = ""
        for sec in sections:
            status = sec.get("status", "not_started")
            ev_count = len(sec.get("evidence_refs", []))
            rows += (
                f"<tr><td>{sec.get('section_number', '')}</td>"
                f"<td>{sec.get('section_title', '')}</td>"
                f'<td class="{_status_css(status)}">{_status_label(status)}</td>'
                f"<td>{sec.get('completion_pct', 0):.0f}%</td>"
                f"<td>{ev_count}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>Bundle Overview</h2>\n'
            f"<p><strong>Overall Completion:</strong> {overall:.0f}%</p>\n"
            "<table><thead><tr><th>Section</th><th>Title</th><th>Status</th>"
            "<th>Completion</th><th>Evidence</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_isae_section(self, sec: Dict[str, Any]) -> str:
        """Render HTML for a single ISAE section."""
        num = sec.get("section_number", 0)
        title = sec.get("section_title", "")
        desc = sec.get("description", "")
        paras = sec.get("isae_paragraphs", "")
        content = f"<h2>Section {num}: {title}</h2>\n"
        if paras:
            content += f"<p><em>ISAE 3410 Reference: {paras}</em></p>\n"
        if desc:
            content += f"<p>{desc}</p>\n"
        # Evidence table
        evidence = sec.get("evidence_refs", [])
        if evidence:
            content += "<h3>Evidence Items</h3>\n<table><thead><tr>"
            content += "<th>ID</th><th>Title</th><th>Type</th><th>Quality</th><th>Hash</th>"
            content += "</tr></thead><tbody>\n"
            for ev in evidence:
                eid = ev.get("evidence_id", "")[:8]
                content += (
                    f"<tr><td><code>{eid}</code></td>"
                    f"<td>{ev.get('title', '')}</td>"
                    f"<td>{ev.get('document_type', '')}</td>"
                    f"<td>{ev.get('quality', 'medium').upper()}</td>"
                    f'<td class="hash-cell">{ev.get("sha256_hash", "")[:16]}...</td></tr>\n'
                )
            content += "</tbody></table>\n"
        # Findings
        findings = sec.get("key_findings", [])
        if findings:
            content += "<h3>Key Findings</h3><ul>"
            for f in findings:
                content += f"<li>{f}</li>"
            content += "</ul>\n"
        # Actions
        actions = sec.get("actions_required", [])
        if actions:
            content += "<h3>Actions Required</h3><ul>"
            for a in actions:
                content += f"<li>{a}</li>"
            content += "</ul>\n"
        return f'<div class="section">\n{content}</div>'

    def _html_xbrl_mapping(self, data: Dict[str, Any]) -> str:
        """Render HTML XBRL mapping."""
        mappings = data.get("xbrl_mappings", [])
        if not mappings:
            return ""
        rows = ""
        for m in mappings:
            rows += (
                f"<tr><td>{m.get('field_name', '')}</td>"
                f"<td><code>{m.get('xbrl_tag', '')}</code></td>"
                f"<td>{m.get('taxonomy', 'ESRS')}</td>"
                f"<td>{m.get('section', '')}</td>"
                f"<td>{m.get('value', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>XBRL Tag Mapping</h2>\n'
            "<table><thead><tr><th>Field</th><th>XBRL Tag</th>"
            "<th>Taxonomy</th><th>Section</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # XBRL RENDERING
    # ==================================================================

    def _render_xbrl(self, data: Dict[str, Any]) -> str:
        """Render ISAE 3410 evidence as XBRL tagged output."""
        provenance = self._compute_provenance(data)
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        ts = self.generated_at.isoformat() if self.generated_at else _utcnow().isoformat()
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"',
            '  xmlns:isae3410="http://greenlang.io/xbrl/isae3410"',
            '  xmlns:link="http://www.xbrl.org/2003/linkbase">',
            "",
            "  <!-- ISAE 3410 Evidence Bundle -->",
            f"  <!-- Company: {company} -->",
            f"  <!-- Period: {period} -->",
            f"  <!-- Generated: {ts} -->",
            f"  <!-- Provenance: {provenance} -->",
            "",
        ]
        for m in data.get("xbrl_mappings", []):
            tag = m.get("xbrl_tag", "")
            value = m.get("value", "")
            dtype = m.get("data_type", "string")
            if tag:
                if dtype == "decimal":
                    lines.append(
                        f'  <isae3410:{tag} decimals="2" unitRef="tCO2e">'
                        f'{value}</isae3410:{tag}>'
                    )
                elif dtype == "date":
                    lines.append(
                        f'  <isae3410:{tag} contextRef="currentPeriod">'
                        f'{value}</isae3410:{tag}>'
                    )
                else:
                    lines.append(
                        f'  <isae3410:{tag}>{value}</isae3410:{tag}>'
                    )
        lines.append("")
        lines.append("</xbrli:xbrl>")
        return "\n".join(lines)

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render evidence bundle as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "isae_3410_evidence_bundle",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "assurance_level": self._get_val(data, "assurance_level", "limited"),
            "verifier_name": self._get_val(data, "verifier_name", ""),
            "overall_completion_pct": data.get("overall_completion_pct", 0),
            "sections": data.get("sections", []),
            "xbrl_mappings": data.get("xbrl_mappings", []),
            "chart_data": {
                "section_completion_bar": self._build_section_bar(data),
                "evidence_quality_pie": self._build_evidence_pie(data),
            },
        }

    def _build_section_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build section completion bar chart data."""
        sections = data.get("sections", [])
        if not sections:
            return {}
        return {
            "labels": [f"Sec {s.get('section_number', '')}" for s in sections],
            "completion": [s.get("completion_pct", 0) for s in sections],
            "statuses": [s.get("status", "not_started") for s in sections],
        }

    def _build_evidence_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build evidence quality pie chart data."""
        sections = data.get("sections", [])
        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "insufficient": 0}
        for sec in sections:
            for ev in sec.get("evidence_refs", []):
                q = ev.get("quality", "medium")
                if q in dist:
                    dist[q] += 1
        if sum(dist.values()) == 0:
            return {}
        return {"labels": list(dist.keys()), "values": list(dist.values())}
