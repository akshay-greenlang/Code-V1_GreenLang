# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Regulatory Filing Package Template
======================================================

Regulatory filing package template covering multi-jurisdiction filing
management, ESEF package validation, filing history, digital signature
status, and submission checklists.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class FilingStatus(str, Enum):
    """Filing status."""
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    OVERDUE = "OVERDUE"


class FormatRequired(str, Enum):
    """Required filing format."""
    ESEF_XHTML = "ESEF_XHTML"
    XBRL = "XBRL"
    PDF = "PDF"
    CSV = "CSV"
    XML = "XML"
    CUSTOM = "CUSTOM"


class ValidationSeverity(str, Enum):
    """Validation error severity."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class JurisdictionFiling(BaseModel):
    """Filing details for a single jurisdiction."""
    country: str = Field(..., description="Country name")
    register_name: str = Field(..., description="Register or authority name")
    format_required: FormatRequired = Field(..., description="Required format")
    filing_deadline: date = Field(..., description="Filing deadline")
    status: FilingStatus = Field(FilingStatus.PENDING, description="Filing status")
    days_remaining: Optional[int] = Field(None, description="Days until deadline")
    contact: Optional[str] = Field(None, description="Contact person")
    notes: Optional[str] = Field(None, description="Additional notes")


class ValidationError(BaseModel):
    """ESEF validation error/warning."""
    code: str = Field(..., description="Error code")
    severity: ValidationSeverity = Field(..., description="Severity level")
    message: str = Field(..., description="Error message")
    location: Optional[str] = Field(None, description="Location in document")


class ESEFPackageStatus(BaseModel):
    """ESEF package validation status."""
    valid: bool = Field(False, description="Overall validation result")
    validation_errors: List[ValidationError] = Field(
        default_factory=list, description="Validation errors/warnings"
    )
    taxonomy_version: str = Field("", description="ESEF taxonomy version used")
    document_count: int = Field(0, ge=0, description="Number of documents in package")
    package_size_mb: Optional[float] = Field(None, description="Package size in MB")
    last_validated: Optional[datetime] = Field(
        None, description="Last validation timestamp"
    )


class FilingRecord(BaseModel):
    """Historical filing record."""
    year: int = Field(..., description="Filing year")
    jurisdiction: str = Field(..., description="Jurisdiction/country")
    filing_date: Optional[date] = Field(None, description="Actual filing date")
    status: FilingStatus = Field(..., description="Filing status")
    reference_number: Optional[str] = Field(None, description="Filing reference")
    format_used: Optional[str] = Field(None, description="Format used")


class SignatureStatus(BaseModel):
    """Digital signature status."""
    signed: bool = Field(False, description="Whether document is signed")
    signer: Optional[str] = Field(None, description="Signer name/role")
    timestamp: Optional[datetime] = Field(None, description="Signing timestamp")
    certificate_valid: bool = Field(False, description="Certificate validity")
    certificate_issuer: Optional[str] = Field(None, description="Certificate issuer")
    certificate_expiry: Optional[date] = Field(None, description="Certificate expiry")


class RegulatoryFilingInput(BaseModel):
    """Complete input for the regulatory filing package."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    filing_jurisdictions: List[JurisdictionFiling] = Field(
        default_factory=list, description="Jurisdictions to file in"
    )
    esef_package: ESEFPackageStatus = Field(
        default_factory=ESEFPackageStatus, description="ESEF package status"
    )
    filing_history: List[FilingRecord] = Field(
        default_factory=list, description="Filing history records"
    )
    digital_signature: SignatureStatus = Field(
        default_factory=SignatureStatus, description="Digital signature status"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _status_badge(status: FilingStatus) -> str:
    """Badge for filing status."""
    return f"[{status.value}]"


def _severity_badge(severity: ValidationSeverity) -> str:
    """Badge for validation severity."""
    return f"[{severity.value}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class RegulatoryFilingPackageTemplate:
    """Generate regulatory filing package report.

    Sections:
        1. Filing Overview Matrix
        2. ESEF Package Validation
        3. Per-Jurisdiction Filing Details
        4. Filing History
        5. Digital Signature Status
        6. Submission Checklist

    Example:
        >>> template = RegulatoryFilingPackageTemplate()
        >>> data = RegulatoryFilingInput(
        ...     organization_name="Acme", reporting_year=2025
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "regulatory_filing_package"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the regulatory filing package template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: RegulatoryFilingInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_filing_overview(data),
            self._md_esef_validation(data),
            self._md_jurisdiction_details(data),
            self._md_filing_history(data),
            self._md_digital_signature(data),
            self._md_submission_checklist(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: RegulatoryFilingInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_filing_overview(data),
            self._html_esef_validation(data),
            self._html_jurisdiction_details(data),
            self._html_filing_history(data),
            self._html_digital_signature(data),
            self._html_submission_checklist(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: RegulatoryFilingInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "filing_jurisdictions": [
                j.model_dump(mode="json") for j in data.filing_jurisdictions
            ],
            "esef_package": data.esef_package.model_dump(mode="json"),
            "filing_history": [
                f.model_dump(mode="json") for f in data.filing_history
            ],
            "digital_signature": data.digital_signature.model_dump(mode="json"),
        }

    def _compute_provenance(self, data: RegulatoryFilingInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: RegulatoryFilingInput) -> str:
        submitted = sum(
            1 for j in data.filing_jurisdictions
            if j.status in (FilingStatus.SUBMITTED, FilingStatus.ACCEPTED)
        )
        total = len(data.filing_jurisdictions)
        return (
            f"# Regulatory Filing Package - {data.organization_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Filings:** {submitted}/{total} submitted\n\n---"
        )

    def _md_filing_overview(self, data: RegulatoryFilingInput) -> str:
        lines = [
            "## 1. Filing Overview Matrix",
            "",
            "| Country | Register | Format | Deadline | Days Left | Status |",
            "|---------|----------|--------|----------|-----------|--------|",
        ]
        for j in sorted(data.filing_jurisdictions, key=lambda x: x.filing_deadline):
            days = str(j.days_remaining) if j.days_remaining is not None else "N/A"
            lines.append(
                f"| {j.country} | {j.register_name} | {j.format_required.value} "
                f"| {j.filing_deadline.isoformat()} | {days} "
                f"| {_status_badge(j.status)} |"
            )
        if not data.filing_jurisdictions:
            lines.append("| - | No filings scheduled | - | - | - | - |")
        return "\n".join(lines)

    def _md_esef_validation(self, data: RegulatoryFilingInput) -> str:
        ep = data.esef_package
        status_text = "VALID" if ep.valid else "INVALID"
        errors = sum(1 for e in ep.validation_errors if e.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for e in ep.validation_errors if e.severity == ValidationSeverity.WARNING)
        size = f"{ep.package_size_mb:.1f} MB" if ep.package_size_mb else "N/A"
        validated_at = ep.last_validated.isoformat() if ep.last_validated else "N/A"
        lines = [
            "## 2. ESEF Package Validation",
            "",
            f"- **Status:** {status_text}",
            f"- **Taxonomy Version:** {ep.taxonomy_version or 'N/A'}",
            f"- **Documents:** {ep.document_count}",
            f"- **Package Size:** {size}",
            f"- **Last Validated:** {validated_at}",
            f"- **Errors:** {errors} | **Warnings:** {warnings}",
        ]
        if ep.validation_errors:
            lines.extend([
                "",
                "### Validation Issues",
                "",
                "| Severity | Code | Message | Location |",
                "|----------|------|---------|----------|",
            ])
            for ve in ep.validation_errors:
                loc = ve.location or "-"
                lines.append(
                    f"| {_severity_badge(ve.severity)} | {ve.code} "
                    f"| {ve.message} | {loc} |"
                )
        return "\n".join(lines)

    def _md_jurisdiction_details(self, data: RegulatoryFilingInput) -> str:
        if not data.filing_jurisdictions:
            return "## 3. Per-Jurisdiction Details\n\nNo jurisdictions configured."
        lines = ["## 3. Per-Jurisdiction Filing Details", ""]
        for i, j in enumerate(data.filing_jurisdictions, 1):
            contact = j.contact or "Not assigned"
            notes = j.notes or "-"
            lines.extend([
                f"### {i}. {j.country} - {j.register_name}",
                "",
                f"- **Format:** {j.format_required.value}",
                f"- **Deadline:** {j.filing_deadline.isoformat()}",
                f"- **Status:** {_status_badge(j.status)}",
                f"- **Contact:** {contact}",
                f"- **Notes:** {notes}",
                "",
            ])
        return "\n".join(lines)

    def _md_filing_history(self, data: RegulatoryFilingInput) -> str:
        if not data.filing_history:
            return "## 4. Filing History\n\nNo filing history available."
        lines = [
            "## 4. Filing History",
            "",
            "| Year | Jurisdiction | Filing Date | Status | Reference | Format |",
            "|------|-------------|-------------|--------|-----------|--------|",
        ]
        for f in sorted(data.filing_history, key=lambda x: x.year, reverse=True):
            fdate = f.filing_date.isoformat() if f.filing_date else "N/A"
            ref = f.reference_number or "-"
            fmt = f.format_used or "-"
            lines.append(
                f"| {f.year} | {f.jurisdiction} | {fdate} "
                f"| {_status_badge(f.status)} | {ref} | {fmt} |"
            )
        return "\n".join(lines)

    def _md_digital_signature(self, data: RegulatoryFilingInput) -> str:
        ds = data.digital_signature
        signed_text = "Yes" if ds.signed else "No"
        cert_valid = "Yes" if ds.certificate_valid else "No"
        signer = ds.signer or "N/A"
        ts = ds.timestamp.isoformat() if ds.timestamp else "N/A"
        issuer = ds.certificate_issuer or "N/A"
        expiry = ds.certificate_expiry.isoformat() if ds.certificate_expiry else "N/A"
        return (
            "## 5. Digital Signature Status\n\n"
            "| Property | Value |\n"
            "|----------|-------|\n"
            f"| Signed | {signed_text} |\n"
            f"| Signer | {signer} |\n"
            f"| Timestamp | {ts} |\n"
            f"| Certificate Valid | {cert_valid} |\n"
            f"| Certificate Issuer | {issuer} |\n"
            f"| Certificate Expiry | {expiry} |"
        )

    def _md_submission_checklist(self, data: RegulatoryFilingInput) -> str:
        checks = [
            ("ESEF package validated", data.esef_package.valid),
            ("Digital signature applied", data.digital_signature.signed),
            ("Certificate valid", data.digital_signature.certificate_valid),
            ("All validation errors resolved",
             not any(e.severity == ValidationSeverity.ERROR
                     for e in data.esef_package.validation_errors)),
        ]
        for j in data.filing_jurisdictions:
            checks.append(
                (f"{j.country} filing prepared",
                 j.status in (FilingStatus.SUBMITTED, FilingStatus.ACCEPTED,
                              FilingStatus.IN_REVIEW))
            )
        lines = ["## 6. Submission Checklist", ""]
        for label, done in checks:
            mark = "[x]" if done else "[ ]"
            lines.append(f"- {mark} {label}")
        all_done = all(done for _, done in checks)
        lines.extend([
            "",
            f"**Overall Readiness:** {'READY TO SUBMIT' if all_done else 'NOT READY'}",
        ])
        return "\n".join(lines)

    def _md_footer(self, data: RegulatoryFilingInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Regulatory Filing Package - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".status-submitted,.status-accepted{color:#1a7f37;font-weight:bold;}\n"
            ".status-pending,.status-in-review{color:#b08800;font-weight:bold;}\n"
            ".status-rejected,.status-overdue{color:#cf222e;font-weight:bold;}\n"
            ".severity-error{color:#cf222e;font-weight:bold;}\n"
            ".severity-warning{color:#b08800;}\n"
            ".severity-info{color:#0969da;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".checklist{list-style:none;padding:0;}\n"
            ".checklist li{padding:0.3rem 0;}\n"
            ".check-pass{color:#1a7f37;}\n"
            ".check-fail{color:#cf222e;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: RegulatoryFilingInput) -> str:
        submitted = sum(
            1 for j in data.filing_jurisdictions
            if j.status in (FilingStatus.SUBMITTED, FilingStatus.ACCEPTED)
        )
        total = len(data.filing_jurisdictions)
        return (
            '<div class="section">\n'
            f"<h1>Regulatory Filing Package &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Filings:</strong> {submitted}/{total} submitted</p>\n"
            "<hr>\n</div>"
        )

    def _html_filing_overview(self, data: RegulatoryFilingInput) -> str:
        rows = []
        for j in sorted(data.filing_jurisdictions, key=lambda x: x.filing_deadline):
            css = f"status-{j.status.value.lower().replace('_', '-')}"
            days = str(j.days_remaining) if j.days_remaining is not None else "N/A"
            rows.append(
                f"<tr><td>{j.country}</td><td>{j.register_name}</td>"
                f"<td>{j.format_required.value}</td>"
                f"<td>{j.filing_deadline.isoformat()}</td><td>{days}</td>"
                f'<td class="{css}">{j.status.value}</td></tr>'
            )
        if not rows:
            rows.append('<tr><td colspan="6">No filings scheduled</td></tr>')
        return (
            '<div class="section">\n<h2>1. Filing Overview Matrix</h2>\n'
            "<table><thead><tr><th>Country</th><th>Register</th><th>Format</th>"
            "<th>Deadline</th><th>Days</th><th>Status</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_esef_validation(self, data: RegulatoryFilingInput) -> str:
        ep = data.esef_package
        status_text = "VALID" if ep.valid else "INVALID"
        status_css = "check-pass" if ep.valid else "check-fail"
        errors = sum(1 for e in ep.validation_errors if e.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for e in ep.validation_errors if e.severity == ValidationSeverity.WARNING)
        parts = [
            '<div class="section">\n<h2>2. ESEF Package Validation</h2>\n',
            f'<div class="metric-card"><div class="metric-value {status_css}">'
            f'{status_text}</div>'
            f'<div class="metric-label">Package Status</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{ep.document_count}</div>'
            f'<div class="metric-label">Documents</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{errors}</div><div class="metric-label">Errors</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{warnings}</div><div class="metric-label">Warnings</div></div>\n',
        ]
        if ep.validation_errors:
            rows = []
            for ve in ep.validation_errors:
                css = f"severity-{ve.severity.value.lower()}"
                loc = ve.location or "-"
                rows.append(
                    f'<tr><td class="{css}">{ve.severity.value}</td>'
                    f"<td>{ve.code}</td><td>{ve.message}</td><td>{loc}</td></tr>"
                )
            parts.append(
                "<h3>Validation Issues</h3>\n"
                "<table><thead><tr><th>Severity</th><th>Code</th>"
                "<th>Message</th><th>Location</th></tr></thead>\n"
                f"<tbody>{''.join(rows)}</tbody></table>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_jurisdiction_details(self, data: RegulatoryFilingInput) -> str:
        if not data.filing_jurisdictions:
            return (
                '<div class="section"><h2>3. Jurisdiction Details</h2>'
                "<p>No jurisdictions configured.</p></div>"
            )
        parts = ['<div class="section">\n<h2>3. Per-Jurisdiction Filing Details</h2>\n']
        for i, j in enumerate(data.filing_jurisdictions, 1):
            contact = j.contact or "Not assigned"
            css = f"status-{j.status.value.lower().replace('_', '-')}"
            parts.append(
                f"<h3>{i}. {j.country} - {j.register_name}</h3>\n"
                f"<ul><li><strong>Format:</strong> {j.format_required.value}</li>\n"
                f"<li><strong>Deadline:</strong> {j.filing_deadline.isoformat()}</li>\n"
                f'<li><strong>Status:</strong> <span class="{css}">{j.status.value}</span></li>\n'
                f"<li><strong>Contact:</strong> {contact}</li></ul>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_filing_history(self, data: RegulatoryFilingInput) -> str:
        if not data.filing_history:
            return (
                '<div class="section"><h2>4. Filing History</h2>'
                "<p>No filing history.</p></div>"
            )
        rows = []
        for f in sorted(data.filing_history, key=lambda x: x.year, reverse=True):
            fdate = f.filing_date.isoformat() if f.filing_date else "N/A"
            ref = f.reference_number or "-"
            fmt = f.format_used or "-"
            css = f"status-{f.status.value.lower().replace('_', '-')}"
            rows.append(
                f"<tr><td>{f.year}</td><td>{f.jurisdiction}</td>"
                f"<td>{fdate}</td>"
                f'<td class="{css}">{f.status.value}</td>'
                f"<td>{ref}</td><td>{fmt}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Filing History</h2>\n'
            "<table><thead><tr><th>Year</th><th>Jurisdiction</th>"
            "<th>Filing Date</th><th>Status</th><th>Reference</th>"
            f"<th>Format</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_digital_signature(self, data: RegulatoryFilingInput) -> str:
        ds = data.digital_signature
        signed_text = "Yes" if ds.signed else "No"
        signed_css = "check-pass" if ds.signed else "check-fail"
        cert_text = "Valid" if ds.certificate_valid else "Invalid"
        cert_css = "check-pass" if ds.certificate_valid else "check-fail"
        signer = ds.signer or "N/A"
        ts = ds.timestamp.isoformat() if ds.timestamp else "N/A"
        rows = [
            f'<tr><td>Signed</td><td class="{signed_css}">{signed_text}</td></tr>',
            f"<tr><td>Signer</td><td>{signer}</td></tr>",
            f"<tr><td>Timestamp</td><td>{ts}</td></tr>",
            f'<tr><td>Certificate</td><td class="{cert_css}">{cert_text}</td></tr>',
        ]
        return (
            '<div class="section">\n<h2>5. Digital Signature Status</h2>\n'
            "<table><thead><tr><th>Property</th><th>Value</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_submission_checklist(self, data: RegulatoryFilingInput) -> str:
        checks = [
            ("ESEF package validated", data.esef_package.valid),
            ("Digital signature applied", data.digital_signature.signed),
            ("Certificate valid", data.digital_signature.certificate_valid),
            ("All errors resolved",
             not any(e.severity == ValidationSeverity.ERROR
                     for e in data.esef_package.validation_errors)),
        ]
        for j in data.filing_jurisdictions:
            checks.append(
                (f"{j.country} filing prepared",
                 j.status in (FilingStatus.SUBMITTED, FilingStatus.ACCEPTED,
                              FilingStatus.IN_REVIEW))
            )
        items = []
        for label, done in checks:
            css = "check-pass" if done else "check-fail"
            mark = "&#10003;" if done else "&#10007;"
            items.append(f'<li class="{css}">{mark} {label}</li>')
        all_done = all(done for _, done in checks)
        ready_css = "check-pass" if all_done else "check-fail"
        ready_text = "READY TO SUBMIT" if all_done else "NOT READY"
        return (
            '<div class="section">\n<h2>6. Submission Checklist</h2>\n'
            f'<ul class="checklist">{"".join(items)}</ul>\n'
            f'<p><strong>Overall Readiness:</strong> '
            f'<span class="{ready_css}">{ready_text}</span></p>\n</div>'
        )

    def _html_footer(self, data: RegulatoryFilingInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
