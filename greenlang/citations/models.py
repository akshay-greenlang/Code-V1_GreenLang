# -*- coding: utf-8 -*-
"""
Citations & Evidence Data Models - AGENT-FOUND-005: Citations & Evidence

Pydantic v2 data models for the Citations & Evidence SDK. These models
are clean SDK versions that mirror the foundation agent enumerations
and models while providing a stable public API.

Models:
    - Enums: CitationType, SourceAuthority, RegulatoryFramework,
             VerificationStatus, EvidenceType, ExportFormat, ChangeType
    - Core: CitationMetadata, Citation, EvidenceItem, EvidencePackage,
            MethodologyReference, RegulatoryRequirement,
            DataSourceAttribution
    - Verification: VerificationRecord
    - Versioning: CitationVersion
    - Audit: ChangeLogEntry

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================


class CitationType(str, Enum):
    """Types of citations supported by the system."""
    EMISSION_FACTOR = "emission_factor"
    REGULATORY = "regulatory"
    METHODOLOGY = "methodology"
    SCIENTIFIC = "scientific"
    COMPANY_DATA = "company_data"
    GUIDANCE = "guidance"
    DATABASE = "database"


class SourceAuthority(str, Enum):
    """Recognized data source authorities."""
    # Emission Factor Sources
    DEFRA = "defra"
    EPA = "epa"
    ECOINVENT = "ecoinvent"
    IPCC = "ipcc"
    GHG_PROTOCOL = "ghg_protocol"
    EXIOBASE = "exiobase"
    CLIMATIQ = "climatiq"
    # Regulatory Bodies
    EU_COMMISSION = "eu_commission"
    SEC = "sec"
    EFRAG = "efrag"
    CARB = "carb"
    # Standards Bodies
    ISO = "iso"
    GRI = "gri"
    SASB = "sasb"
    CDP = "cdp"
    # Internal
    INTERNAL = "internal"
    SUPPLIER = "supplier"
    # Other
    OTHER = "other"


class RegulatoryFramework(str, Enum):
    """Regulatory frameworks for compliance tracking."""
    CSRD = "csrd"
    CBAM = "cbam"
    EUDR = "eudr"
    SB253 = "sb253"
    SB261 = "sb261"
    SEC_CLIMATE = "sec_climate"
    TCFD = "tcfd"
    TNFD = "tnfd"


class VerificationStatus(str, Enum):
    """Status of citation verification."""
    VERIFIED = "verified"
    PENDING = "pending"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    UNVERIFIED = "unverified"
    INVALID = "invalid"


class EvidenceType(str, Enum):
    """Types of evidence that can be packaged."""
    CALCULATION = "calculation"
    DATA_POINT = "data_point"
    METHODOLOGY = "methodology"
    ASSUMPTION = "assumption"
    VALIDATION = "validation"
    AUDIT_TRAIL = "audit_trail"


class ExportFormat(str, Enum):
    """Supported export formats for citations."""
    BIBTEX = "bibtex"
    JSON = "json"
    CSL = "csl"


class ChangeType(str, Enum):
    """Types of changes to citations and evidence."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VERIFY = "verify"
    SUPERSEDE = "supersede"


# =============================================================================
# Utility
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Core Data Models
# =============================================================================


class CitationMetadata(BaseModel):
    """Metadata for a citation."""

    title: str = Field(..., description="Title of the cited work")
    authors: List[str] = Field(
        default_factory=list, description="Authors of the work",
    )
    publication_date: Optional[date] = Field(
        None, description="Publication date",
    )
    version: Optional[str] = Field(
        None, description="Version number (e.g., '2024')",
    )
    edition: Optional[str] = Field(
        None, description="Edition if applicable",
    )
    publisher: Optional[str] = Field(
        None, description="Publisher or organization",
    )
    url: Optional[str] = Field(
        None, description="URL for online access",
    )
    doi: Optional[str] = Field(
        None, description="Digital Object Identifier",
    )
    isbn: Optional[str] = Field(
        None, description="ISBN for books",
    )
    issn: Optional[str] = Field(
        None, description="ISSN for journals",
    )
    page_numbers: Optional[str] = Field(
        None, description="Page range (e.g., '15-23')",
    )
    chapter: Optional[str] = Field(
        None, description="Chapter name or number",
    )
    section: Optional[str] = Field(
        None, description="Section reference",
    )
    table_reference: Optional[str] = Field(
        None, description="Table number/name",
    )

    model_config = {"extra": "forbid"}

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, v: Optional[str]) -> Optional[str]:
        """Validate DOI format if provided."""
        if v is None:
            return v
        if not re.match(r"^10\.\d{4,}/[^\s]+$", v):
            raise ValueError(f"Invalid DOI format: {v}")
        return v


class Citation(BaseModel):
    """Complete citation record for zero-hallucination compliance.

    Every data point used in calculations must be traceable to a Citation.
    """

    citation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this citation",
    )
    citation_type: CitationType = Field(
        ..., description="Type of citation",
    )
    source_authority: SourceAuthority = Field(
        ..., description="Authority/organization providing the data",
    )
    metadata: CitationMetadata = Field(
        ..., description="Citation metadata",
    )

    # Versioning
    effective_date: date = Field(
        ..., description="Date from which this citation is effective",
    )
    expiration_date: Optional[date] = Field(
        None, description="Date when this citation expires",
    )
    superseded_by: Optional[str] = Field(
        None, description="Citation ID that supersedes this one",
    )
    supersedes: Optional[str] = Field(
        None, description="Citation ID that this one supersedes",
    )

    # Verification
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Verification status",
    )
    verified_at: Optional[datetime] = Field(
        None, description="When verification was performed",
    )
    verified_by: Optional[str] = Field(
        None, description="Who verified this citation",
    )

    # Regulatory linkage
    regulatory_frameworks: List[RegulatoryFramework] = Field(
        default_factory=list,
        description="Regulatory frameworks this citation supports",
    )

    # Content
    abstract: Optional[str] = Field(
        None, description="Brief description of the content",
    )
    key_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key values/factors from this citation",
    )
    notes: Optional[str] = Field(
        None, description="Additional notes",
    )

    # Audit trail
    created_at: datetime = Field(
        default_factory=_utcnow, description="When this record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="When this record was last updated",
    )
    created_by: Optional[str] = Field(
        None, description="User who created this record",
    )

    # Provenance hash
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of citation content",
    )

    model_config = {"extra": "forbid"}

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of citation content for provenance."""
        content = {
            "citation_type": self.citation_type.value,
            "source_authority": self.source_authority.value,
            "metadata": self.metadata.model_dump(mode="json"),
            "effective_date": self.effective_date.isoformat(),
            "key_values": self.key_values,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def is_valid(self, reference_date: Optional[date] = None) -> bool:
        """Check if citation is valid as of a given date."""
        if reference_date is None:
            reference_date = date.today()
        if self.effective_date > reference_date:
            return False
        if self.expiration_date and self.expiration_date < reference_date:
            return False
        if self.verification_status in (
            VerificationStatus.INVALID,
            VerificationStatus.EXPIRED,
        ):
            return False
        return True

    def to_bibtex(self) -> str:
        """Export citation to BibTeX format."""
        entry_type = self._get_bibtex_type()
        bibtex_id = self._generate_bibtex_id()
        fields = []

        if self.metadata.title:
            fields.append(f"  title = {{{self.metadata.title}}}")
        if self.metadata.authors:
            authors_str = " and ".join(self.metadata.authors)
            fields.append(f"  author = {{{authors_str}}}")
        if self.metadata.publication_date:
            fields.append(
                f"  year = {{{self.metadata.publication_date.year}}}",
            )
        if self.metadata.publisher:
            fields.append(f"  publisher = {{{self.metadata.publisher}}}")
        if self.metadata.url:
            fields.append(f"  url = {{{self.metadata.url}}}")
        if self.metadata.doi:
            fields.append(f"  doi = {{{self.metadata.doi}}}")
        if self.metadata.isbn:
            fields.append(f"  isbn = {{{self.metadata.isbn}}}")
        if self.metadata.version:
            fields.append(f"  edition = {{{self.metadata.version}}}")

        fields_str = ",\n".join(fields)
        return f"@{entry_type}{{{bibtex_id},\n{fields_str}\n}}"

    def to_csl(self) -> Dict[str, Any]:
        """Export citation to CSL-JSON format (Zotero/Mendeley compatible)."""
        csl: Dict[str, Any] = {
            "id": self.citation_id,
            "type": self._get_csl_type(),
            "title": self.metadata.title,
        }

        if self.metadata.authors:
            csl["author"] = [
                {"literal": name} for name in self.metadata.authors
            ]
        if self.metadata.publication_date:
            csl["issued"] = {
                "date-parts": [[
                    self.metadata.publication_date.year,
                    self.metadata.publication_date.month,
                    self.metadata.publication_date.day,
                ]],
            }
        if self.metadata.publisher:
            csl["publisher"] = self.metadata.publisher
        if self.metadata.url:
            csl["URL"] = self.metadata.url
        if self.metadata.doi:
            csl["DOI"] = self.metadata.doi
        if self.metadata.isbn:
            csl["ISBN"] = self.metadata.isbn
        if self.metadata.issn:
            csl["ISSN"] = self.metadata.issn
        if self.metadata.page_numbers:
            csl["page"] = self.metadata.page_numbers
        if self.metadata.version:
            csl["edition"] = self.metadata.version
        if self.abstract:
            csl["abstract"] = self.abstract

        return csl

    def _get_bibtex_type(self) -> str:
        """Determine BibTeX entry type."""
        type_mapping = {
            CitationType.EMISSION_FACTOR: "techreport",
            CitationType.REGULATORY: "misc",
            CitationType.METHODOLOGY: "manual",
            CitationType.SCIENTIFIC: "article",
            CitationType.COMPANY_DATA: "misc",
            CitationType.GUIDANCE: "techreport",
            CitationType.DATABASE: "misc",
        }
        return type_mapping.get(self.citation_type, "misc")

    def _get_csl_type(self) -> str:
        """Determine CSL type string."""
        type_mapping = {
            CitationType.EMISSION_FACTOR: "report",
            CitationType.REGULATORY: "legislation",
            CitationType.METHODOLOGY: "standard",
            CitationType.SCIENTIFIC: "article-journal",
            CitationType.COMPANY_DATA: "dataset",
            CitationType.GUIDANCE: "report",
            CitationType.DATABASE: "dataset",
        }
        return type_mapping.get(self.citation_type, "document")

    def _generate_bibtex_id(self) -> str:
        """Generate a BibTeX-friendly ID."""
        if self.metadata.authors:
            surname = self.metadata.authors[0].split()[-1].lower()
        else:
            surname = self.source_authority.value

        year = ""
        if self.metadata.publication_date:
            year = str(self.metadata.publication_date.year)

        bibtex_id = re.sub(r"[^a-z0-9]", "", f"{surname}{year}")
        return bibtex_id or self.citation_id[:8]


class EvidenceItem(BaseModel):
    """A single piece of evidence in an evidence package."""

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evidence",
    )
    evidence_type: EvidenceType = Field(
        ..., description="Type of evidence",
    )
    description: str = Field(
        ..., description="Description of what this evidence demonstrates",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="The evidence data",
    )
    citation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of citations supporting this evidence",
    )
    source_system: Optional[str] = Field(
        None, description="System that produced this evidence",
    )
    source_agent: Optional[str] = Field(
        None, description="Agent that produced this evidence",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="When this evidence was recorded",
    )
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of evidence content",
    )

    model_config = {"extra": "forbid"}

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        content = {
            "evidence_type": self.evidence_type.value,
            "description": self.description,
            "data": self.data,
            "citation_ids": sorted(self.citation_ids),
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


class EvidencePackage(BaseModel):
    """Complete evidence package for audit-ready documentation.

    Bundles all evidence, citations, and calculations for a specific
    calculation or data point.
    """

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this package",
    )
    name: str = Field(
        ..., description="Name of this evidence package",
    )
    description: str = Field(
        default="", description="Description of what this package documents",
    )

    # Content
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items in this package",
    )
    citation_ids: List[str] = Field(
        default_factory=list,
        description="Citation IDs included in this package",
    )

    # Context
    calculation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context for calculations (inputs, parameters)",
    )
    calculation_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results of calculations",
    )

    # Regulatory
    regulatory_frameworks: List[RegulatoryFramework] = Field(
        default_factory=list,
        description="Regulatory frameworks this package supports",
    )
    compliance_notes: Optional[str] = Field(
        None, description="Notes on regulatory compliance",
    )

    # Audit metadata
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this package was created",
    )
    created_by: Optional[str] = Field(
        None, description="User who created this package",
    )
    finalized_at: Optional[datetime] = Field(
        None, description="When this package was finalized",
    )

    # Integrity
    package_hash: Optional[str] = Field(
        None, description="SHA-256 hash of entire package",
    )

    model_config = {"extra": "forbid"}

    def calculate_package_hash(self) -> str:
        """Calculate SHA-256 hash of entire package."""
        content = {
            "name": self.name,
            "evidence_items": [
                item.calculate_content_hash()
                for item in self.evidence_items
            ],
            "citation_ids": sorted(self.citation_ids),
            "calculation_context": self.calculation_context,
            "calculation_result": self.calculation_result,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    @property
    def is_finalized(self) -> bool:
        """Return True if the package has been finalized."""
        return self.package_hash is not None


class MethodologyReference(BaseModel):
    """Reference to a calculation methodology."""

    reference_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    name: str = Field(
        ..., description="Name of the methodology",
    )
    standard: str = Field(
        ..., description="Standard (e.g., 'GHG Protocol', 'ISO 14064-1')",
    )
    version: str = Field(
        ..., description="Version of the standard",
    )
    section: Optional[str] = Field(
        None, description="Specific section reference",
    )
    description: str = Field(
        default="", description="Description of the methodology",
    )
    citation_id: Optional[str] = Field(
        None, description="ID of the supporting citation",
    )
    scope_1_applicable: bool = Field(default=False)
    scope_2_applicable: bool = Field(default=False)
    scope_3_applicable: bool = Field(default=False)
    applicable_categories: List[str] = Field(
        default_factory=list,
        description="Applicable Scope 3 categories",
    )
    formula_id: Optional[str] = Field(
        None, description="ID of the formula in the formula registry",
    )

    model_config = {"extra": "forbid"}


class RegulatoryRequirement(BaseModel):
    """A regulatory requirement for compliance tracking."""

    requirement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    framework: RegulatoryFramework = Field(
        ..., description="Regulatory framework",
    )
    article: Optional[str] = Field(
        None, description="Article number (e.g., 'Article 29b')",
    )
    requirement_text: str = Field(
        ..., description="Text of the requirement",
    )
    citation_id: Optional[str] = Field(
        None, description="ID of the source citation",
    )
    effective_date: date = Field(
        ..., description="When requirement becomes effective",
    )
    compliance_deadline: Optional[date] = Field(
        None, description="Deadline for compliance",
    )
    applies_to_scope_1: bool = Field(default=False)
    applies_to_scope_2: bool = Field(default=False)
    applies_to_scope_3: bool = Field(default=False)
    compliance_status: Optional[str] = Field(
        None, description="Current compliance status",
    )
    compliance_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence package IDs demonstrating compliance",
    )

    model_config = {"extra": "forbid"}


class DataSourceAttribution(BaseModel):
    """Attribution for a data source."""

    attribution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    source_authority: SourceAuthority = Field(
        ..., description="Source authority",
    )
    dataset_name: str = Field(
        ..., description="Name of the dataset",
    )
    dataset_version: str = Field(
        ..., description="Version of the dataset",
    )
    citation_id: Optional[str] = Field(
        None, description="ID of the source citation",
    )
    extracted_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values extracted from this source",
    )
    extraction_date: datetime = Field(
        default_factory=_utcnow,
        description="When data was extracted",
    )
    extracted_by: Optional[str] = Field(
        None, description="Who extracted the data",
    )
    valid_from: date = Field(
        ..., description="Start of validity period",
    )
    valid_until: Optional[date] = Field(
        None, description="End of validity period",
    )

    model_config = {"extra": "forbid"}


class VerificationRecord(BaseModel):
    """Record of a citation verification check."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique verification record ID",
    )
    citation_id: str = Field(
        ..., description="Citation that was verified",
    )
    status: VerificationStatus = Field(
        ..., description="Result of verification",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="When the check was performed",
    )
    checked_by: str = Field(
        default="system", description="Who performed the check",
    )
    checks_performed: List[str] = Field(
        default_factory=list,
        description="List of checks that were run",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues found during verification",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class CitationVersion(BaseModel):
    """A single version snapshot of a citation."""

    version_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique version ID",
    )
    version_number: int = Field(
        ..., ge=1, description="Sequential version number",
    )
    citation_id: str = Field(
        ..., description="ID of the citation this version belongs to",
    )
    snapshot: Dict[str, Any] = Field(
        ..., description="Full citation snapshot at this version",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp",
    )
    created_by: str = Field(
        ..., description="User who created this version",
    )
    change_reason: str = Field(
        ..., description="Reason for the change",
    )
    change_type: ChangeType = Field(
        ..., description="Type of change",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )
    parent_version_id: Optional[str] = Field(
        None, description="Previous version ID",
    )

    model_config = {"extra": "forbid"}


class ChangeLogEntry(BaseModel):
    """Audit log entry for citation and evidence changes."""

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique log ID",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Change timestamp",
    )
    user_id: str = Field(
        ..., description="User who made the change",
    )
    change_type: ChangeType = Field(
        ..., description="Type of change",
    )
    entity_type: str = Field(
        ..., description="Type of entity changed (citation, package, etc.)",
    )
    entity_id: str = Field(
        ..., description="ID of the changed entity",
    )
    old_value: Optional[Any] = Field(
        None, description="Previous value",
    )
    new_value: Optional[Any] = Field(
        None, description="New value",
    )
    change_reason: str = Field(
        ..., description="Reason for change",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit",
    )

    model_config = {"extra": "forbid"}


__all__ = [
    # Enumerations
    "CitationType",
    "SourceAuthority",
    "RegulatoryFramework",
    "VerificationStatus",
    "EvidenceType",
    "ExportFormat",
    "ChangeType",
    # Core models
    "CitationMetadata",
    "Citation",
    "EvidenceItem",
    "EvidencePackage",
    "MethodologyReference",
    "RegulatoryRequirement",
    "DataSourceAttribution",
    # Verification
    "VerificationRecord",
    # Versioning
    "CitationVersion",
    # Audit
    "ChangeLogEntry",
]
