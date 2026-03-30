# -*- coding: utf-8 -*-
"""
EvidencePackageIndex - Evidence Package Index for PACK-048.

Generates a complete evidence package inventory organised by scope and
category, with evidence quality distribution (pie chart data),
completeness percentages by scope (S1/S2/S3) and category, highlighted
missing evidence items, SHA-256 hashes per evidence item, and package
version and generation timestamps.

Regulatory References:
    - ISAE 3410 para 53-56: Obtaining evidence
    - ISO 14064-3 clause 6.3: Evidence requirements
    - GHG Protocol Corporate Standard (Chapter 10)
    - CSRD / ESRS: Audit trail and documentation requirements
    - PCAF Standard: Data quality scoring methodology

Sections:
    1. Package Overview (version, timestamp, totals)
    2. Evidence Inventory by Scope and Category
    3. Quality Distribution (pie chart data)
    4. Completeness by Scope (S1/S2/S3)
    5. Missing Evidence Items
    6. Evidence Hash Register
    7. Provenance Footer

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class EvidenceQuality(str, Enum):
    """Evidence quality classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"

class EvidenceStatus(str, Enum):
    """Evidence item status."""
    AVAILABLE = "available"
    MISSING = "missing"
    PARTIAL = "partial"
    EXPIRED = "expired"

class ScopeLabel(str, Enum):
    """GHG Scope labels."""
    SCOPE_1 = "Scope 1"
    SCOPE_2 = "Scope 2"
    SCOPE_3 = "Scope 3"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """Single evidence item in the package."""
    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence unique ID")
    title: str = Field(..., description="Evidence item title")
    description: str = Field("", description="Evidence item description")
    scope: str = Field("", description="GHG scope (Scope 1, Scope 2, Scope 3)")
    category: str = Field("", description="Emissions category or sub-category")
    source: str = Field("", description="Evidence source (e.g., ERP, invoice)")
    document_type: str = Field("", description="Document type (e.g., invoice, meter reading)")
    quality: EvidenceQuality = Field(EvidenceQuality.MEDIUM, description="Quality rating")
    status: EvidenceStatus = Field(EvidenceStatus.AVAILABLE, description="Availability status")
    file_path: str = Field("", description="Path or reference to evidence file")
    sha256_hash: str = Field("", description="SHA-256 hash of evidence file")
    collected_date: Optional[str] = Field(None, description="Collection date (ISO)")
    expiry_date: Optional[str] = Field(None, description="Expiry date (ISO)")
    reviewer: str = Field("", description="Reviewer name")
    notes: str = Field("", description="Additional notes")

class ScopeCompleteness(BaseModel):
    """Completeness summary for a single scope."""
    scope: str = Field(..., description="Scope label")
    total_required: int = Field(0, ge=0, description="Total required evidence items")
    total_available: int = Field(0, ge=0, description="Total available items")
    total_missing: int = Field(0, ge=0, description="Total missing items")
    total_partial: int = Field(0, ge=0, description="Total partial items")
    completeness_pct: float = Field(0.0, ge=0, le=100, description="Completeness %")
    categories: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-category breakdown"
    )

class PackageMetadata(BaseModel):
    """Evidence package metadata."""
    package_id: str = Field(default_factory=_new_uuid, description="Package unique ID")
    package_version: str = Field("1.0.0", description="Package version")
    generated_at: Optional[str] = Field(None, description="Generation timestamp (ISO)")
    total_items: int = Field(0, ge=0, description="Total evidence items")
    total_available: int = Field(0, ge=0, description="Total available items")
    total_missing: int = Field(0, ge=0, description="Total missing items")
    overall_completeness_pct: float = Field(0.0, ge=0, le=100, description="Overall completeness %")

class EvidencePackageInput(BaseModel):
    """Complete input model for EvidencePackageIndex."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    package_metadata: Optional[PackageMetadata] = Field(
        None, description="Package metadata"
    )
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list, description="All evidence items"
    )
    scope_completeness: List[ScopeCompleteness] = Field(
        default_factory=list, description="Completeness by scope"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _quality_label(quality: str) -> str:
    """Return display label for quality."""
    return quality.upper()

def _status_label(status: str) -> str:
    """Return display label for status."""
    return status.upper()

def _quality_css(quality: str) -> str:
    """Return CSS class for quality."""
    mapping = {
        "high": "qual-high",
        "medium": "qual-medium",
        "low": "qual-low",
        "insufficient": "qual-insufficient",
    }
    return mapping.get(quality, "qual-medium")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class EvidencePackageIndex:
    """
    Evidence package index template for PACK-048.

    Renders a complete evidence inventory with quality distribution,
    completeness percentages by scope and category, missing items
    highlight, and SHA-256 hashes per evidence item. All outputs
    include SHA-256 provenance hashing for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = EvidencePackageIndex()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EvidencePackageIndex."""
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
        """Render evidence package index as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render evidence package index as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render evidence package index as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
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
            self._md_package_overview(data),
            self._md_evidence_inventory(data),
            self._md_quality_distribution(data),
            self._md_scope_completeness(data),
            self._md_missing_items(data),
            self._md_hash_register(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Evidence Package Index - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_package_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown package overview."""
        meta = data.get("package_metadata") or {}
        lines = [
            "## 1. Package Overview",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Package ID | `{meta.get('package_id', 'N/A')}` |",
            f"| Version | {meta.get('package_version', '1.0.0')} |",
            f"| Generated At | {meta.get('generated_at', 'N/A')} |",
            f"| Total Items | {meta.get('total_items', 0)} |",
            f"| Available | {meta.get('total_available', 0)} |",
            f"| Missing | {meta.get('total_missing', 0)} |",
            f"| Completeness | {meta.get('overall_completeness_pct', 0):.1f}% |",
        ]
        return "\n".join(lines)

    def _md_evidence_inventory(self, data: Dict[str, Any]) -> str:
        """Render Markdown evidence inventory table."""
        items = data.get("evidence_items", [])
        if not items:
            return "## 2. Evidence Inventory\n\nNo evidence items available."
        lines = [
            "## 2. Evidence Inventory",
            "",
            "| ID | Title | Scope | Category | Source | Quality | Status |",
            "|----|-------|-------|----------|--------|---------|--------|",
        ]
        for item in items:
            eid = item.get("evidence_id", "")[:8]
            lines.append(
                f"| `{eid}` | "
                f"{item.get('title', '')} | "
                f"{item.get('scope', '')} | "
                f"{item.get('category', '')} | "
                f"{item.get('source', '')} | "
                f"**{_quality_label(item.get('quality', 'medium'))}** | "
                f"{_status_label(item.get('status', 'available'))} |"
            )
        return "\n".join(lines)

    def _md_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality distribution summary."""
        items = data.get("evidence_items", [])
        if not items:
            return ""
        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "insufficient": 0}
        for item in items:
            q = item.get("quality", "medium")
            if q in dist:
                dist[q] += 1
        total = sum(dist.values()) or 1
        lines = [
            "## 3. Evidence Quality Distribution",
            "",
            "| Quality | Count | Percentage |",
            "|---------|-------|------------|",
        ]
        for quality, count in dist.items():
            pct = (count / total) * 100
            lines.append(f"| {quality.upper()} | {count} | {pct:.1f}% |")
        return "\n".join(lines)

    def _md_scope_completeness(self, data: Dict[str, Any]) -> str:
        """Render Markdown completeness by scope."""
        scopes = data.get("scope_completeness", [])
        if not scopes:
            return ""
        lines = [
            "## 4. Completeness by Scope",
            "",
            "| Scope | Required | Available | Missing | Partial | Completeness |",
            "|-------|----------|-----------|---------|---------|--------------|",
        ]
        for s in scopes:
            lines.append(
                f"| {s.get('scope', '')} | "
                f"{s.get('total_required', 0)} | "
                f"{s.get('total_available', 0)} | "
                f"{s.get('total_missing', 0)} | "
                f"{s.get('total_partial', 0)} | "
                f"{s.get('completeness_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _md_missing_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown missing evidence items."""
        items = data.get("evidence_items", [])
        missing = [i for i in items if i.get("status") in ("missing", "expired")]
        if not missing:
            return "## 5. Missing Evidence Items\n\nNo missing items identified."
        lines = [
            "## 5. Missing Evidence Items",
            "",
            "| Title | Scope | Category | Status | Notes |",
            "|-------|-------|----------|--------|-------|",
        ]
        for item in missing:
            lines.append(
                f"| {item.get('title', '')} | "
                f"{item.get('scope', '')} | "
                f"{item.get('category', '')} | "
                f"**{_status_label(item.get('status', 'missing'))}** | "
                f"{item.get('notes', '')} |"
            )
        return "\n".join(lines)

    def _md_hash_register(self, data: Dict[str, Any]) -> str:
        """Render Markdown evidence hash register."""
        items = data.get("evidence_items", [])
        hashed = [i for i in items if i.get("sha256_hash")]
        if not hashed:
            return ""
        lines = [
            "## 6. Evidence Hash Register",
            "",
            "| Title | SHA-256 Hash |",
            "|-------|-------------|",
        ]
        for item in hashed:
            lines.append(
                f"| {item.get('title', '')} | "
                f"`{item.get('sha256_hash', '')}` |"
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
            self._html_package_overview(data),
            self._html_evidence_inventory(data),
            self._html_quality_distribution(data),
            self._html_scope_completeness(data),
            self._html_missing_items(data),
            self._html_hash_register(data),
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
            f"<title>Evidence Package Index - {company}</title>\n"
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
            ".qual-high{color:#2a9d8f;font-weight:700;}\n"
            ".qual-medium{color:#e9c46a;font-weight:700;}\n"
            ".qual-low{color:#e76f51;font-weight:700;}\n"
            ".qual-insufficient{color:#d62828;font-weight:700;}\n"
            ".status-missing{color:#e76f51;font-weight:700;}\n"
            ".status-available{color:#2a9d8f;}\n"
            ".hash-cell{font-family:monospace;font-size:0.75rem;}\n"
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
            f"<h1>Evidence Package Index &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_package_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML package overview."""
        meta = data.get("package_metadata") or {}
        rows = (
            f"<tr><td>Package ID</td><td><code>{meta.get('package_id', 'N/A')}</code></td></tr>\n"
            f"<tr><td>Version</td><td>{meta.get('package_version', '1.0.0')}</td></tr>\n"
            f"<tr><td>Generated At</td><td>{meta.get('generated_at', 'N/A')}</td></tr>\n"
            f"<tr><td>Total Items</td><td>{meta.get('total_items', 0)}</td></tr>\n"
            f"<tr><td>Available</td><td>{meta.get('total_available', 0)}</td></tr>\n"
            f"<tr><td>Missing</td><td>{meta.get('total_missing', 0)}</td></tr>\n"
            f"<tr><td>Completeness</td><td>{meta.get('overall_completeness_pct', 0):.1f}%</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>1. Package Overview</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_evidence_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML evidence inventory."""
        items = data.get("evidence_items", [])
        if not items:
            return ""
        rows = ""
        for item in items:
            eid = item.get("evidence_id", "")[:8]
            quality = item.get("quality", "medium")
            status = item.get("status", "available")
            rows += (
                f"<tr><td><code>{eid}</code></td>"
                f"<td>{item.get('title', '')}</td>"
                f"<td>{item.get('scope', '')}</td>"
                f"<td>{item.get('category', '')}</td>"
                f"<td>{item.get('source', '')}</td>"
                f'<td class="{_quality_css(quality)}">{_quality_label(quality)}</td>'
                f"<td>{_status_label(status)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Evidence Inventory</h2>\n'
            "<table><thead><tr><th>ID</th><th>Title</th><th>Scope</th>"
            "<th>Category</th><th>Source</th><th>Quality</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML quality distribution."""
        items = data.get("evidence_items", [])
        if not items:
            return ""
        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "insufficient": 0}
        for item in items:
            q = item.get("quality", "medium")
            if q in dist:
                dist[q] += 1
        total = sum(dist.values()) or 1
        rows = ""
        for quality, count in dist.items():
            pct = (count / total) * 100
            rows += (
                f'<tr><td class="{_quality_css(quality)}">{quality.upper()}</td>'
                f"<td>{count}</td><td>{pct:.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Evidence Quality Distribution</h2>\n'
            "<table><thead><tr><th>Quality</th><th>Count</th>"
            "<th>Percentage</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope_completeness(self, data: Dict[str, Any]) -> str:
        """Render HTML scope completeness."""
        scopes = data.get("scope_completeness", [])
        if not scopes:
            return ""
        rows = ""
        for s in scopes:
            rows += (
                f"<tr><td><strong>{s.get('scope', '')}</strong></td>"
                f"<td>{s.get('total_required', 0)}</td>"
                f"<td>{s.get('total_available', 0)}</td>"
                f"<td>{s.get('total_missing', 0)}</td>"
                f"<td>{s.get('total_partial', 0)}</td>"
                f"<td>{s.get('completeness_pct', 0):.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Completeness by Scope</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Required</th><th>Available</th>"
            "<th>Missing</th><th>Partial</th><th>Completeness</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_missing_items(self, data: Dict[str, Any]) -> str:
        """Render HTML missing evidence items."""
        items = data.get("evidence_items", [])
        missing = [i for i in items if i.get("status") in ("missing", "expired")]
        if not missing:
            return ""
        rows = ""
        for item in missing:
            rows += (
                f'<tr><td>{item.get("title", "")}</td>'
                f'<td>{item.get("scope", "")}</td>'
                f'<td>{item.get("category", "")}</td>'
                f'<td class="status-missing">{_status_label(item.get("status", "missing"))}</td>'
                f'<td>{item.get("notes", "")}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>5. Missing Evidence Items</h2>\n'
            "<table><thead><tr><th>Title</th><th>Scope</th><th>Category</th>"
            "<th>Status</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_hash_register(self, data: Dict[str, Any]) -> str:
        """Render HTML evidence hash register."""
        items = data.get("evidence_items", [])
        hashed = [i for i in items if i.get("sha256_hash")]
        if not hashed:
            return ""
        rows = ""
        for item in hashed:
            rows += (
                f"<tr><td>{item.get('title', '')}</td>"
                f'<td class="hash-cell">{item.get("sha256_hash", "")}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>6. Evidence Hash Register</h2>\n'
            "<table><thead><tr><th>Title</th><th>SHA-256 Hash</th></tr></thead>\n"
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
        """Render evidence package as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "evidence_package_index",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "package_metadata": data.get("package_metadata"),
            "evidence_items": data.get("evidence_items", []),
            "scope_completeness": data.get("scope_completeness", []),
            "chart_data": {
                "quality_pie": self._build_quality_pie(data),
                "scope_bar": self._build_scope_bar_chart(data),
                "completeness_gauge": self._build_completeness_gauge(data),
            },
        }

    def _build_quality_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build quality distribution pie chart data."""
        items = data.get("evidence_items", [])
        if not items:
            return {}
        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "insufficient": 0}
        for item in items:
            q = item.get("quality", "medium")
            if q in dist:
                dist[q] += 1
        return {
            "labels": list(dist.keys()),
            "values": list(dist.values()),
            "colors": ["#2a9d8f", "#e9c46a", "#e76f51", "#d62828"],
        }

    def _build_scope_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scope completeness bar chart data."""
        scopes = data.get("scope_completeness", [])
        if not scopes:
            return {}
        return {
            "labels": [s.get("scope", "") for s in scopes],
            "available": [s.get("total_available", 0) for s in scopes],
            "missing": [s.get("total_missing", 0) for s in scopes],
            "partial": [s.get("total_partial", 0) for s in scopes],
        }

    def _build_completeness_gauge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall completeness gauge chart data."""
        meta = data.get("package_metadata") or {}
        return {
            "value": meta.get("overall_completeness_pct", 0),
            "min": 0,
            "max": 100,
        }
