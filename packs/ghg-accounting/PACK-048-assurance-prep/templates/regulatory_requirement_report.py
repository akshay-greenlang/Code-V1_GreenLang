# -*- coding: utf-8 -*-
"""
RegulatoryRequirementReport - Regulatory Requirement Report for PACK-048.

Generates a regulatory requirement report for GHG assurance with a
jurisdiction-by-jurisdiction requirements table, compliance status per
jurisdiction (compliant/gap/not_applicable), timeline visualisation
(requirement effective dates), gap analysis summary with action items,
and multi-jurisdiction overlap analysis.

Regulatory References:
    - CSRD / ESRS: EU mandatory assurance requirements
    - SEC Climate Disclosure Rule: US GHG assurance requirements
    - ISSB IFRS S2: Global sustainability disclosure standards
    - UK Companies Act / Streamlined Energy & Carbon Reporting
    - Singapore Exchange Listing Rules (SGX)
    - Australian Climate-Related Financial Disclosure

Sections:
    1. Jurisdiction Requirements Table
    2. Compliance Status Matrix
    3. Timeline Visualisation (effective dates)
    4. Gap Analysis Summary
    5. Multi-Jurisdiction Overlap
    6. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

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


class ComplianceStatus(str, Enum):
    """Compliance status per jurisdiction."""
    COMPLIANT = "compliant"
    GAP = "gap"
    NOT_APPLICABLE = "not_applicable"
    IN_PROGRESS = "in_progress"


class AssuranceLevel(str, Enum):
    """Required assurance level."""
    LIMITED = "limited"
    REASONABLE = "reasonable"
    NONE = "none"
    PHASED = "phased"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class JurisdictionRequirement(BaseModel):
    """Requirement for a single jurisdiction."""
    jurisdiction: str = Field(..., description="Jurisdiction name (e.g., EU, US, UK)")
    regulation_name: str = Field("", description="Regulation name (e.g., CSRD)")
    effective_date: Optional[str] = Field(None, description="Effective date (ISO)")
    assurance_level_required: AssuranceLevel = Field(
        AssuranceLevel.LIMITED, description="Required assurance level"
    )
    scope_coverage: str = Field("", description="Scopes covered (e.g., S1+S2, S1+S2+S3)")
    reporting_standard: str = Field("", description="Required standard (e.g., ESRS, SEC)")
    assurance_standard: str = Field("", description="Required assurance standard")
    applicability_criteria: str = Field("", description="Who must comply (size, listing, etc.)")
    phase_in_notes: str = Field("", description="Phase-in timeline notes")
    compliance_status: ComplianceStatus = Field(
        ComplianceStatus.IN_PROGRESS, description="Current compliance status"
    )
    gap_description: str = Field("", description="Gap description (if gap)")
    action_items: List[str] = Field(default_factory=list, description="Action items to close gap")


class TimelineEntry(BaseModel):
    """Timeline entry for requirement effective dates."""
    jurisdiction: str = Field(..., description="Jurisdiction")
    regulation: str = Field("", description="Regulation name")
    milestone: str = Field("", description="Milestone description")
    date: Optional[str] = Field(None, description="Date (ISO)")
    milestone_type: str = Field("effective", description="Type: effective/phase_in/deadline")


class OverlapEntry(BaseModel):
    """Multi-jurisdiction overlap entry."""
    requirement_area: str = Field(..., description="Requirement area")
    jurisdictions: List[str] = Field(default_factory=list, description="Jurisdictions with this requirement")
    common_standard: str = Field("", description="Common standard if any")
    divergences: List[str] = Field(default_factory=list, description="Key divergences")
    harmonisation_notes: str = Field("", description="Harmonisation / streamlining notes")


class GapActionItem(BaseModel):
    """Gap analysis action item."""
    gap_id: str = Field("", description="Gap identifier")
    jurisdiction: str = Field("", description="Jurisdiction")
    requirement: str = Field("", description="Requirement description")
    gap_description: str = Field("", description="Gap description")
    action: str = Field("", description="Action to close")
    priority: str = Field("medium", description="Priority (high/medium/low)")
    owner: str = Field("", description="Responsible party")
    target_date: Optional[str] = Field(None, description="Target date (ISO)")
    status: str = Field("open", description="Status")


class RegulatoryRequirementInput(BaseModel):
    """Complete input model for RegulatoryRequirementReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    requirements: List[JurisdictionRequirement] = Field(
        default_factory=list, description="Jurisdiction requirements"
    )
    timeline: List[TimelineEntry] = Field(
        default_factory=list, description="Timeline entries"
    )
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Multi-jurisdiction overlaps"
    )
    gap_actions: List[GapActionItem] = Field(
        default_factory=list, description="Gap action items"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _status_label(status: str) -> str:
    """Return display label for compliance status."""
    return status.replace("_", " ").upper()


def _status_css(status: str) -> str:
    """Return CSS class for compliance status."""
    mapping = {
        "compliant": "st-compliant",
        "gap": "st-gap",
        "not_applicable": "st-na",
        "in_progress": "st-progress",
    }
    return mapping.get(status, "st-progress")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class RegulatoryRequirementReport:
    """
    Regulatory requirement report template for PACK-048.

    Renders a jurisdiction-by-jurisdiction requirements table, compliance
    matrix, timeline, gap analysis, and overlap analysis. All outputs
    include SHA-256 provenance hashing for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = RegulatoryRequirementReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryRequirementReport."""
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
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render regulatory requirements as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory requirements as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory requirements as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
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
            self._md_requirements_table(data),
            self._md_compliance_matrix(data),
            self._md_timeline(data),
            self._md_gap_analysis(data),
            self._md_overlap_analysis(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Regulatory Requirement Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_requirements_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown jurisdiction requirements table."""
        reqs = data.get("requirements", [])
        if not reqs:
            return "## 1. Jurisdiction Requirements\n\nNo requirements defined."
        lines = [
            "## 1. Jurisdiction Requirements",
            "",
            "| Jurisdiction | Regulation | Effective | Assurance Level | Scopes | Standard | Status |",
            "|-------------|-----------|-----------|----------------|--------|----------|--------|",
        ]
        for r in reqs:
            level = r.get("assurance_level_required", "limited").replace("_", " ").title()
            status = r.get("compliance_status", "in_progress")
            lines.append(
                f"| {r.get('jurisdiction', '')} | "
                f"{r.get('regulation_name', '')} | "
                f"{r.get('effective_date', '-')} | "
                f"{level} | "
                f"{r.get('scope_coverage', '')} | "
                f"{r.get('assurance_standard', '')} | "
                f"**{_status_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_compliance_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown compliance status matrix."""
        reqs = data.get("requirements", [])
        if not reqs:
            return ""
        compliant = sum(1 for r in reqs if r.get("compliance_status") == "compliant")
        gap = sum(1 for r in reqs if r.get("compliance_status") == "gap")
        progress = sum(1 for r in reqs if r.get("compliance_status") == "in_progress")
        na = sum(1 for r in reqs if r.get("compliance_status") == "not_applicable")
        lines = [
            "## 2. Compliance Status Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Compliant | {compliant} |",
            f"| Gap | {gap} |",
            f"| In Progress | {progress} |",
            f"| Not Applicable | {na} |",
            f"| **Total** | **{len(reqs)}** |",
        ]
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render Markdown timeline."""
        entries = data.get("timeline", [])
        if not entries:
            return ""
        lines = [
            "## 3. Requirements Timeline",
            "",
            "| Date | Jurisdiction | Regulation | Milestone | Type |",
            "|------|-------------|-----------|-----------|------|",
        ]
        sorted_entries = sorted(entries, key=lambda e: e.get("date", ""))
        for e in sorted_entries:
            lines.append(
                f"| {e.get('date', '-')} | "
                f"{e.get('jurisdiction', '')} | "
                f"{e.get('regulation', '')} | "
                f"{e.get('milestone', '')} | "
                f"{e.get('milestone_type', 'effective').title()} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis."""
        actions = data.get("gap_actions", [])
        if not actions:
            return ""
        lines = [
            "## 4. Gap Analysis & Action Items",
            "",
            "| ID | Jurisdiction | Requirement | Gap | Action | Priority | Owner | Target | Status |",
            "|----|-------------|-------------|-----|--------|----------|-------|--------|--------|",
        ]
        for a in actions:
            lines.append(
                f"| {a.get('gap_id', '')} | "
                f"{a.get('jurisdiction', '')} | "
                f"{a.get('requirement', '')} | "
                f"{a.get('gap_description', '')} | "
                f"{a.get('action', '')} | "
                f"**{a.get('priority', 'medium').upper()}** | "
                f"{a.get('owner', '')} | "
                f"{a.get('target_date', '-')} | "
                f"{a.get('status', 'open')} |"
            )
        return "\n".join(lines)

    def _md_overlap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown multi-jurisdiction overlap analysis."""
        overlaps = data.get("overlaps", [])
        if not overlaps:
            return ""
        lines = [
            "## 5. Multi-Jurisdiction Overlap Analysis",
            "",
            "| Area | Jurisdictions | Common Standard | Divergences |",
            "|------|---------------|----------------|-------------|",
        ]
        for o in overlaps:
            jurisdictions = ", ".join(o.get("jurisdictions", []))
            divergences = "; ".join(o.get("divergences", [])[:3]) or "-"
            lines.append(
                f"| {o.get('requirement_area', '')} | "
                f"{jurisdictions} | "
                f"{o.get('common_standard', '-')} | "
                f"{divergences} |"
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
            self._html_requirements_table(data),
            self._html_compliance_matrix(data),
            self._html_timeline(data),
            self._html_gap_analysis(data),
            self._html_overlap_analysis(data),
            self._html_footer(data),
        ]
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
            f"<title>Regulatory Requirements - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".st-compliant{color:#2a9d8f;font-weight:700;}\n"
            ".st-gap{color:#e76f51;font-weight:700;}\n"
            ".st-na{color:#888;}\n"
            ".st-progress{color:#e9c46a;font-weight:700;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Regulatory Requirement Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_requirements_table(self, data: Dict[str, Any]) -> str:
        """Render HTML jurisdiction requirements table."""
        reqs = data.get("requirements", [])
        if not reqs:
            return ""
        rows = ""
        for r in reqs:
            level = r.get("assurance_level_required", "limited").replace("_", " ").title()
            status = r.get("compliance_status", "in_progress")
            rows += (
                f"<tr><td>{r.get('jurisdiction', '')}</td>"
                f"<td>{r.get('regulation_name', '')}</td>"
                f"<td>{r.get('effective_date', '-')}</td>"
                f"<td>{level}</td>"
                f"<td>{r.get('scope_coverage', '')}</td>"
                f"<td>{r.get('assurance_standard', '')}</td>"
                f'<td class="{_status_css(status)}">{_status_label(status)}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>1. Jurisdiction Requirements</h2>\n'
            "<table><thead><tr><th>Jurisdiction</th><th>Regulation</th>"
            "<th>Effective</th><th>Level</th><th>Scopes</th>"
            "<th>Standard</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_compliance_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status summary."""
        reqs = data.get("requirements", [])
        if not reqs:
            return ""
        compliant = sum(1 for r in reqs if r.get("compliance_status") == "compliant")
        gap = sum(1 for r in reqs if r.get("compliance_status") == "gap")
        progress = sum(1 for r in reqs if r.get("compliance_status") == "in_progress")
        na = sum(1 for r in reqs if r.get("compliance_status") == "not_applicable")
        rows = (
            f'<tr><td class="st-compliant">Compliant</td><td>{compliant}</td></tr>\n'
            f'<tr><td class="st-gap">Gap</td><td>{gap}</td></tr>\n'
            f'<tr><td class="st-progress">In Progress</td><td>{progress}</td></tr>\n'
            f'<tr><td class="st-na">Not Applicable</td><td>{na}</td></tr>\n'
            f"<tr><td><strong>Total</strong></td><td><strong>{len(reqs)}</strong></td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>2. Compliance Status Summary</h2>\n'
            "<table><thead><tr><th>Status</th><th>Count</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML timeline."""
        entries = data.get("timeline", [])
        if not entries:
            return ""
        sorted_entries = sorted(entries, key=lambda e: e.get("date", ""))
        rows = ""
        for e in sorted_entries:
            rows += (
                f"<tr><td>{e.get('date', '-')}</td>"
                f"<td>{e.get('jurisdiction', '')}</td>"
                f"<td>{e.get('regulation', '')}</td>"
                f"<td>{e.get('milestone', '')}</td>"
                f"<td>{e.get('milestone_type', 'effective').title()}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Requirements Timeline</h2>\n'
            "<table><thead><tr><th>Date</th><th>Jurisdiction</th><th>Regulation</th>"
            "<th>Milestone</th><th>Type</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        actions = data.get("gap_actions", [])
        if not actions:
            return ""
        rows = ""
        for a in actions:
            rows += (
                f"<tr><td>{a.get('gap_id', '')}</td>"
                f"<td>{a.get('jurisdiction', '')}</td>"
                f"<td>{a.get('gap_description', '')}</td>"
                f"<td>{a.get('action', '')}</td>"
                f"<td>{a.get('priority', 'medium').upper()}</td>"
                f"<td>{a.get('owner', '')}</td>"
                f"<td>{a.get('target_date', '-')}</td>"
                f"<td>{a.get('status', 'open')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Gap Analysis & Action Items</h2>\n'
            "<table><thead><tr><th>ID</th><th>Jurisdiction</th><th>Gap</th>"
            "<th>Action</th><th>Priority</th><th>Owner</th>"
            "<th>Target</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_overlap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML overlap analysis."""
        overlaps = data.get("overlaps", [])
        if not overlaps:
            return ""
        rows = ""
        for o in overlaps:
            jurisdictions = ", ".join(o.get("jurisdictions", []))
            divergences = "; ".join(o.get("divergences", [])[:3]) or "-"
            rows += (
                f"<tr><td>{o.get('requirement_area', '')}</td>"
                f"<td>{jurisdictions}</td>"
                f"<td>{o.get('common_standard', '-')}</td>"
                f"<td>{divergences}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Multi-Jurisdiction Overlap</h2>\n'
            "<table><thead><tr><th>Area</th><th>Jurisdictions</th>"
            "<th>Common Standard</th><th>Divergences</th></tr></thead>\n"
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
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory requirements as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "regulatory_requirement_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "requirements": data.get("requirements", []),
            "timeline": data.get("timeline", []),
            "overlaps": data.get("overlaps", []),
            "gap_actions": data.get("gap_actions", []),
            "chart_data": {
                "compliance_pie": self._build_compliance_pie(data),
                "timeline_gantt": self._build_timeline_gantt(data),
                "overlap_matrix": self._build_overlap_matrix(data),
            },
        }

    def _build_compliance_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build compliance status pie chart data."""
        reqs = data.get("requirements", [])
        if not reqs:
            return {}
        dist: Dict[str, int] = {"compliant": 0, "gap": 0, "in_progress": 0, "not_applicable": 0}
        for r in reqs:
            st = r.get("compliance_status", "in_progress")
            if st in dist:
                dist[st] += 1
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def _build_timeline_gantt(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build timeline Gantt chart data."""
        entries = data.get("timeline", [])
        return [
            {
                "jurisdiction": e.get("jurisdiction", ""),
                "regulation": e.get("regulation", ""),
                "date": e.get("date", ""),
                "milestone": e.get("milestone", ""),
                "type": e.get("milestone_type", "effective"),
            }
            for e in entries
        ]

    def _build_overlap_matrix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overlap matrix data."""
        overlaps = data.get("overlaps", [])
        if not overlaps:
            return {}
        return {
            "areas": [o.get("requirement_area", "") for o in overlaps],
            "jurisdiction_lists": [o.get("jurisdictions", []) for o in overlaps],
        }
