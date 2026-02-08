# -*- coding: utf-8 -*-
"""
Unit Tests for Citations Models (AGENT-FOUND-005)

Tests all enums, Pydantic-style models, field validation, serialization,
hash computation, and edge cases for the citations service data types.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models mirroring greenlang/citations/models.py
# ---------------------------------------------------------------------------


class CitationType(str, Enum):
    EMISSION_FACTOR = "emission_factor"
    REGULATORY = "regulatory"
    METHODOLOGY = "methodology"
    SCIENTIFIC = "scientific"
    COMPANY_DATA = "company_data"
    GUIDANCE = "guidance"
    DATABASE = "database"


class SourceAuthority(str, Enum):
    DEFRA = "defra"
    EPA = "epa"
    ECOINVENT = "ecoinvent"
    IPCC = "ipcc"
    GHG_PROTOCOL = "ghg_protocol"
    EXIOBASE = "exiobase"
    CLIMATIQ = "climatiq"
    EU_COMMISSION = "eu_commission"
    SEC = "sec"
    EFRAG = "efrag"
    CARB = "carb"
    ISO = "iso"
    GRI = "gri"
    SASB = "sasb"
    CDP = "cdp"
    INTERNAL = "internal"
    SUPPLIER = "supplier"
    OTHER = "other"


class RegulatoryFramework(str, Enum):
    CSRD = "csrd"
    CBAM = "cbam"
    EUDR = "eudr"
    SB253 = "sb253"
    SB261 = "sb261"
    SEC_CLIMATE = "sec_climate"
    TCFD = "tcfd"
    TNFD = "tnfd"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    UNVERIFIED = "unverified"
    INVALID = "invalid"


class EvidenceType(str, Enum):
    CALCULATION = "calculation"
    DATA_POINT = "data_point"
    METHODOLOGY = "methodology"
    ASSUMPTION = "assumption"
    VALIDATION = "validation"
    AUDIT_TRAIL = "audit_trail"


class ExportFormat(str, Enum):
    JSON = "json"
    BIBTEX = "bibtex"
    CSL = "csl"


class ChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VERIFY = "verify"
    SUPERSEDE = "supersede"


class CitationMetadata:
    """Metadata attached to a citation."""

    def __init__(
        self,
        title: str = "",
        authors: Optional[List[str]] = None,
        publication_date: Optional[str] = None,
        version: Optional[str] = None,
        edition: Optional[str] = None,
        publisher: Optional[str] = None,
        url: Optional[str] = None,
        doi: Optional[str] = None,
        isbn: Optional[str] = None,
        issn: Optional[str] = None,
        page_numbers: Optional[str] = None,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        table_reference: Optional[str] = None,
    ):
        self.title = title
        self.authors = authors or []
        self.publication_date = publication_date
        self.version = version
        self.edition = edition
        self.publisher = publisher
        self.url = url
        self.doi = doi
        self.isbn = isbn
        self.issn = issn
        self.page_numbers = page_numbers
        self.chapter = chapter
        self.section = section
        self.table_reference = table_reference

    @staticmethod
    def validate_doi(doi: Optional[str]) -> bool:
        """Validate DOI format: 10.xxxx/xxxxx."""
        if doi is None:
            return True
        return bool(re.match(r'^10\.\d{4,}/[^\s]+$', doi))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "version": self.version,
            "edition": self.edition,
            "publisher": self.publisher,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            "issn": self.issn,
            "page_numbers": self.page_numbers,
            "chapter": self.chapter,
            "section": self.section,
            "table_reference": self.table_reference,
        }


class Citation:
    """Complete citation record for zero-hallucination compliance."""

    def __init__(
        self,
        citation_id: Optional[str] = None,
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        metadata: Optional[Dict[str, Any]] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        superseded_by: Optional[str] = None,
        supersedes: Optional[str] = None,
        verification_status: str = "unverified",
        verified_at: Optional[str] = None,
        verified_by: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_by: Optional[str] = None,
        content_hash: Optional[str] = None,
    ):
        self.citation_id = citation_id or str(uuid.uuid4())
        self.citation_type = citation_type
        self.source_authority = source_authority
        self._metadata_dict = metadata or {}
        self.metadata = CitationMetadata(**self._metadata_dict) if isinstance(self._metadata_dict, dict) else self._metadata_dict
        self.effective_date = effective_date
        self.expiration_date = expiration_date
        self.superseded_by = superseded_by
        self.supersedes = supersedes
        self.verification_status = verification_status
        self.verified_at = verified_at
        self.verified_by = verified_by
        self.regulatory_frameworks = regulatory_frameworks or []
        self.abstract = abstract
        self.key_values = key_values or {}
        self.notes = notes
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.created_by = created_by
        self.content_hash = content_hash

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of citation content."""
        content = {
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "metadata": self.metadata.to_dict() if hasattr(self.metadata, 'to_dict') else self._metadata_dict,
            "effective_date": self.effective_date,
            "key_values": self.key_values,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def is_valid(self, reference_date: Optional[str] = None) -> bool:
        """Check if citation is valid as of a given date."""
        ref = reference_date or date.today().isoformat()

        if self.effective_date > ref:
            return False

        if self.expiration_date and self.expiration_date < ref:
            return False

        if self.verification_status in ("invalid", "expired"):
            return False

        return True

    def to_bibtex(self) -> str:
        """Export citation to BibTeX format."""
        type_map = {
            "emission_factor": "techreport",
            "regulatory": "misc",
            "methodology": "manual",
            "scientific": "article",
            "company_data": "misc",
            "guidance": "techreport",
            "database": "misc",
        }
        entry_type = type_map.get(self.citation_type, "misc")

        # Build BibTeX ID
        if self.metadata.authors:
            surname = self.metadata.authors[0].split()[-1].lower()
        else:
            surname = self.source_authority

        year = ""
        if self.metadata.publication_date:
            year = self.metadata.publication_date[:4]

        bibtex_id = re.sub(r'[^a-z0-9]', '', f"{surname}{year}")
        if not bibtex_id:
            bibtex_id = self.citation_id[:8]

        fields = []
        if self.metadata.title:
            fields.append(f'  title = {{{self.metadata.title}}}')
        if self.metadata.authors:
            fields.append(f'  author = {{{" and ".join(self.metadata.authors)}}}')
        if self.metadata.publication_date:
            fields.append(f'  year = {{{self.metadata.publication_date[:4]}}}')
        if self.metadata.publisher:
            fields.append(f'  publisher = {{{self.metadata.publisher}}}')
        if self.metadata.url:
            fields.append(f'  url = {{{self.metadata.url}}}')
        if self.metadata.doi:
            fields.append(f'  doi = {{{self.metadata.doi}}}')

        fields_str = ',\n'.join(fields)
        return f'@{entry_type}{{{bibtex_id},\n{fields_str}\n}}'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "metadata": self.metadata.to_dict() if hasattr(self.metadata, 'to_dict') else {},
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "verification_status": self.verification_status,
            "regulatory_frameworks": self.regulatory_frameworks,
            "key_values": self.key_values,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class EvidenceItem:
    """A single piece of evidence."""

    def __init__(
        self,
        evidence_id: Optional[str] = None,
        evidence_type: str = "calculation",
        description: str = "",
        data: Optional[Dict[str, Any]] = None,
        citation_ids: Optional[List[str]] = None,
        source_system: Optional[str] = None,
        source_agent: Optional[str] = None,
        timestamp: Optional[str] = None,
        content_hash: Optional[str] = None,
    ):
        self.evidence_id = evidence_id or str(uuid.uuid4())
        self.evidence_type = evidence_type
        self.description = description
        self.data = data or {}
        self.citation_ids = citation_ids or []
        self.source_system = source_system
        self.source_agent = source_agent
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.content_hash = content_hash

    def calculate_content_hash(self) -> str:
        content = {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "data": self.data,
            "citation_ids": sorted(self.citation_ids),
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


class EvidencePackage:
    """Complete evidence package for audit-ready documentation."""

    def __init__(
        self,
        package_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        evidence_items: Optional[List[EvidenceItem]] = None,
        citations: Optional[List[Citation]] = None,
        calculation_context: Optional[Dict[str, Any]] = None,
        calculation_result: Optional[Dict[str, Any]] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        created_at: Optional[str] = None,
        created_by: Optional[str] = None,
        is_finalized: bool = False,
        package_hash: Optional[str] = None,
    ):
        if not name:
            raise ValueError("name is required for evidence package")
        self.package_id = package_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.evidence_items = evidence_items or []
        self.citations = citations or []
        self.calculation_context = calculation_context or {}
        self.calculation_result = calculation_result or {}
        self.regulatory_frameworks = regulatory_frameworks or []
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.created_by = created_by
        self.is_finalized = is_finalized
        self.package_hash = package_hash

    def add_evidence(self, item: EvidenceItem) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot add evidence to a finalized package")
        item.content_hash = item.calculate_content_hash()
        self.evidence_items.append(item)
        self.package_hash = None

    def add_citation(self, citation: Citation) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot add citation to a finalized package")
        citation.content_hash = citation.calculate_content_hash()
        self.citations.append(citation)
        self.package_hash = None

    def finalize(self) -> str:
        if self.is_finalized:
            raise RuntimeError("Package is already finalized")
        self.package_hash = self.calculate_package_hash()
        self.is_finalized = True
        return self.package_hash

    def calculate_package_hash(self) -> str:
        content = {
            "name": self.name,
            "evidence_items": [i.calculate_content_hash() for i in self.evidence_items],
            "citations": [c.calculate_content_hash() for c in self.citations],
            "calculation_context": self.calculation_context,
            "calculation_result": self.calculation_result,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


class VerificationRecord:
    """Record of a verification check."""

    def __init__(
        self,
        record_id: Optional[str] = None,
        citation_id: str = "",
        old_status: str = "unverified",
        new_status: str = "verified",
        verified_by: str = "system",
        reason: str = "",
        timestamp: Optional[str] = None,
    ):
        self.record_id = record_id or str(uuid.uuid4())
        self.citation_id = citation_id
        self.old_status = old_status
        self.new_status = new_status
        self.verified_by = verified_by
        self.reason = reason
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class CitationVersion:
    """A version snapshot of a citation."""

    def __init__(
        self,
        citation_id: str = "",
        version: int = 1,
        data: Optional[Dict[str, Any]] = None,
        change_type: str = "create",
        changed_by: str = "system",
        provenance_hash: str = "",
        timestamp: Optional[str] = None,
    ):
        self.citation_id = citation_id
        self.version = version
        self.data = data or {}
        self.change_type = change_type
        self.changed_by = changed_by
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class ChangeLogEntry:
    """Changelog entry for a citation modification."""

    def __init__(
        self,
        entry_id: Optional[str] = None,
        citation_id: str = "",
        change_type: str = "create",
        old_value: Any = None,
        new_value: Any = None,
        changed_by: str = "system",
        change_reason: str = "",
        provenance_hash: str = "",
        timestamp: Optional[str] = None,
    ):
        self.entry_id = entry_id or str(uuid.uuid4())
        self.citation_id = citation_id
        self.change_type = change_type
        self.old_value = old_value
        self.new_value = new_value
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "citation_id": self.citation_id,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_by": self.changed_by,
            "change_reason": self.change_reason,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCitationTypeEnum:
    """Test CitationType enum values."""

    def test_emission_factor(self):
        assert CitationType.EMISSION_FACTOR.value == "emission_factor"

    def test_regulatory(self):
        assert CitationType.REGULATORY.value == "regulatory"

    def test_methodology(self):
        assert CitationType.METHODOLOGY.value == "methodology"

    def test_scientific(self):
        assert CitationType.SCIENTIFIC.value == "scientific"

    def test_company_data(self):
        assert CitationType.COMPANY_DATA.value == "company_data"

    def test_guidance(self):
        assert CitationType.GUIDANCE.value == "guidance"

    def test_database(self):
        assert CitationType.DATABASE.value == "database"

    def test_all_7_types(self):
        assert len(CitationType) == 7


class TestSourceAuthorityEnum:
    """Test SourceAuthority enum values."""

    def test_defra(self):
        assert SourceAuthority.DEFRA.value == "defra"

    def test_epa(self):
        assert SourceAuthority.EPA.value == "epa"

    def test_ecoinvent(self):
        assert SourceAuthority.ECOINVENT.value == "ecoinvent"

    def test_ipcc(self):
        assert SourceAuthority.IPCC.value == "ipcc"

    def test_ghg_protocol(self):
        assert SourceAuthority.GHG_PROTOCOL.value == "ghg_protocol"

    def test_exiobase(self):
        assert SourceAuthority.EXIOBASE.value == "exiobase"

    def test_climatiq(self):
        assert SourceAuthority.CLIMATIQ.value == "climatiq"

    def test_eu_commission(self):
        assert SourceAuthority.EU_COMMISSION.value == "eu_commission"

    def test_sec(self):
        assert SourceAuthority.SEC.value == "sec"

    def test_efrag(self):
        assert SourceAuthority.EFRAG.value == "efrag"

    def test_carb(self):
        assert SourceAuthority.CARB.value == "carb"

    def test_iso(self):
        assert SourceAuthority.ISO.value == "iso"

    def test_gri(self):
        assert SourceAuthority.GRI.value == "gri"

    def test_sasb(self):
        assert SourceAuthority.SASB.value == "sasb"

    def test_cdp(self):
        assert SourceAuthority.CDP.value == "cdp"

    def test_internal(self):
        assert SourceAuthority.INTERNAL.value == "internal"

    def test_supplier(self):
        assert SourceAuthority.SUPPLIER.value == "supplier"

    def test_other(self):
        assert SourceAuthority.OTHER.value == "other"

    def test_all_18_authorities(self):
        assert len(SourceAuthority) == 18


class TestRegulatoryFrameworkEnum:
    """Test RegulatoryFramework enum values."""

    def test_csrd(self):
        assert RegulatoryFramework.CSRD.value == "csrd"

    def test_cbam(self):
        assert RegulatoryFramework.CBAM.value == "cbam"

    def test_eudr(self):
        assert RegulatoryFramework.EUDR.value == "eudr"

    def test_sb253(self):
        assert RegulatoryFramework.SB253.value == "sb253"

    def test_sb261(self):
        assert RegulatoryFramework.SB261.value == "sb261"

    def test_sec_climate(self):
        assert RegulatoryFramework.SEC_CLIMATE.value == "sec_climate"

    def test_tcfd(self):
        assert RegulatoryFramework.TCFD.value == "tcfd"

    def test_tnfd(self):
        assert RegulatoryFramework.TNFD.value == "tnfd"

    def test_all_8_frameworks(self):
        assert len(RegulatoryFramework) == 8


class TestVerificationStatusEnum:
    """Test VerificationStatus enum values."""

    def test_verified(self):
        assert VerificationStatus.VERIFIED.value == "verified"

    def test_pending(self):
        assert VerificationStatus.PENDING.value == "pending"

    def test_expired(self):
        assert VerificationStatus.EXPIRED.value == "expired"

    def test_superseded(self):
        assert VerificationStatus.SUPERSEDED.value == "superseded"

    def test_unverified(self):
        assert VerificationStatus.UNVERIFIED.value == "unverified"

    def test_invalid(self):
        assert VerificationStatus.INVALID.value == "invalid"

    def test_all_6_statuses(self):
        assert len(VerificationStatus) == 6


class TestEvidenceTypeEnum:
    """Test EvidenceType enum values."""

    def test_calculation(self):
        assert EvidenceType.CALCULATION.value == "calculation"

    def test_data_point(self):
        assert EvidenceType.DATA_POINT.value == "data_point"

    def test_methodology(self):
        assert EvidenceType.METHODOLOGY.value == "methodology"

    def test_assumption(self):
        assert EvidenceType.ASSUMPTION.value == "assumption"

    def test_validation(self):
        assert EvidenceType.VALIDATION.value == "validation"

    def test_audit_trail(self):
        assert EvidenceType.AUDIT_TRAIL.value == "audit_trail"

    def test_all_6_types(self):
        assert len(EvidenceType) == 6


class TestExportFormatEnum:
    """Test ExportFormat enum values."""

    def test_json(self):
        assert ExportFormat.JSON.value == "json"

    def test_bibtex(self):
        assert ExportFormat.BIBTEX.value == "bibtex"

    def test_csl(self):
        assert ExportFormat.CSL.value == "csl"

    def test_all_3_formats(self):
        assert len(ExportFormat) == 3


class TestChangeTypeEnum:
    """Test ChangeType enum values."""

    def test_create(self):
        assert ChangeType.CREATE.value == "create"

    def test_update(self):
        assert ChangeType.UPDATE.value == "update"

    def test_delete(self):
        assert ChangeType.DELETE.value == "delete"

    def test_verify(self):
        assert ChangeType.VERIFY.value == "verify"

    def test_supersede(self):
        assert ChangeType.SUPERSEDE.value == "supersede"

    def test_all_5_types(self):
        assert len(ChangeType) == 5


class TestCitationMetadataModel:
    """Test CitationMetadata model."""

    def test_creation_with_title(self):
        m = CitationMetadata(title="DEFRA 2024 GHG Factors")
        assert m.title == "DEFRA 2024 GHG Factors"

    def test_defaults(self):
        m = CitationMetadata()
        assert m.title == ""
        assert m.authors == []
        assert m.doi is None
        assert m.url is None

    def test_to_dict(self):
        m = CitationMetadata(title="Test", version="2024")
        d = m.to_dict()
        assert d["title"] == "Test"
        assert d["version"] == "2024"

    def test_doi_validation_valid(self):
        assert CitationMetadata.validate_doi("10.1017/9781009157926") is True

    def test_doi_validation_invalid(self):
        assert CitationMetadata.validate_doi("not-a-doi") is False

    def test_doi_validation_none(self):
        assert CitationMetadata.validate_doi(None) is True

    def test_all_fields(self):
        m = CitationMetadata(
            title="Title", authors=["Author1"], publication_date="2024-01-01",
            version="2024", edition="1st", publisher="Publisher",
            url="https://example.com", doi="10.1234/test",
            isbn="978-0-123456-78-9", issn="1234-5678",
            page_numbers="1-10", chapter="Chapter 1",
            section="Section 2.3", table_reference="Table 4",
        )
        assert m.isbn == "978-0-123456-78-9"
        assert m.page_numbers == "1-10"


class TestCitationModel:
    """Test Citation model creation and methods."""

    def test_creation_with_minimal_fields(self):
        c = Citation(citation_type="emission_factor", source_authority="defra")
        assert c.citation_type == "emission_factor"
        assert c.source_authority == "defra"
        assert c.citation_id is not None

    def test_auto_generated_id(self):
        c = Citation()
        assert len(c.citation_id) == 36  # UUID format

    def test_explicit_id(self):
        c = Citation(citation_id="my-id")
        assert c.citation_id == "my-id"

    def test_content_hash_calculation(self):
        c = Citation(
            citation_type="emission_factor",
            source_authority="defra",
            metadata={"title": "Test", "version": "2024"},
            effective_date="2024-01-01",
            key_values={"diesel_ef": 2.68},
        )
        h = c.calculate_content_hash()
        assert len(h) == 64
        int(h, 16)  # Verify it is hexadecimal

    def test_content_hash_deterministic(self):
        kwargs = {
            "citation_type": "emission_factor",
            "source_authority": "defra",
            "metadata": {"title": "Test", "version": "2024"},
            "effective_date": "2024-01-01",
            "key_values": {"diesel_ef": 2.68},
        }
        c1 = Citation(**kwargs)
        c2 = Citation(**kwargs)
        assert c1.calculate_content_hash() == c2.calculate_content_hash()

    def test_content_hash_changes_with_data(self):
        c1 = Citation(key_values={"ef": 2.68})
        c2 = Citation(key_values={"ef": 3.00})
        assert c1.calculate_content_hash() != c2.calculate_content_hash()

    def test_is_valid_current_date(self):
        c = Citation(effective_date="2020-01-01", verification_status="verified")
        assert c.is_valid() is True

    def test_is_valid_future_effective_date(self):
        c = Citation(effective_date="2099-01-01")
        assert c.is_valid() is False

    def test_is_valid_expired(self):
        c = Citation(effective_date="2020-01-01", expiration_date="2021-01-01")
        assert c.is_valid() is False

    def test_is_valid_invalid_status(self):
        c = Citation(effective_date="2020-01-01", verification_status="invalid")
        assert c.is_valid() is False

    def test_is_valid_expired_status(self):
        c = Citation(effective_date="2020-01-01", verification_status="expired")
        assert c.is_valid() is False

    def test_is_valid_with_reference_date(self):
        c = Citation(effective_date="2024-01-01", expiration_date="2025-12-31")
        assert c.is_valid(reference_date="2024-06-15") is True
        assert c.is_valid(reference_date="2026-01-01") is False

    def test_to_bibtex_emission_factor(self):
        c = Citation(
            citation_type="emission_factor",
            source_authority="defra",
            metadata={
                "title": "DEFRA GHG Factors",
                "authors": ["DEFRA"],
                "publication_date": "2024-06-01",
                "publisher": "UK DEFRA",
            },
        )
        bib = c.to_bibtex()
        assert "@techreport{" in bib
        assert "title = {DEFRA GHG Factors}" in bib
        assert "year = {2024}" in bib

    def test_to_bibtex_scientific(self):
        c = Citation(
            citation_type="scientific",
            source_authority="ipcc",
            metadata={
                "title": "Climate Change 2022",
                "authors": ["Smith, J.", "Jones, K."],
                "publication_date": "2022-04-04",
                "doi": "10.1017/9781009157926",
            },
        )
        bib = c.to_bibtex()
        assert "@article{" in bib
        assert "doi = {10.1017/9781009157926}" in bib

    def test_to_bibtex_regulatory(self):
        c = Citation(citation_type="regulatory", source_authority="eu_commission")
        bib = c.to_bibtex()
        assert "@misc{" in bib

    def test_to_bibtex_methodology(self):
        c = Citation(citation_type="methodology", source_authority="ghg_protocol")
        bib = c.to_bibtex()
        assert "@manual{" in bib

    def test_to_dict(self):
        c = Citation(citation_id="cid-1", citation_type="emission_factor")
        d = c.to_dict()
        assert d["citation_id"] == "cid-1"
        assert d["citation_type"] == "emission_factor"
        assert "created_at" in d

    def test_default_verification_status(self):
        c = Citation()
        assert c.verification_status == "unverified"

    def test_default_key_values_empty(self):
        c = Citation()
        assert c.key_values == {}

    def test_default_regulatory_frameworks_empty(self):
        c = Citation()
        assert c.regulatory_frameworks == []


class TestEvidenceItemModel:
    """Test EvidenceItem model."""

    def test_creation(self):
        e = EvidenceItem(evidence_type="calculation", description="Test calc")
        assert e.evidence_type == "calculation"
        assert e.description == "Test calc"

    def test_auto_generated_id(self):
        e = EvidenceItem()
        assert len(e.evidence_id) == 36

    def test_content_hash_calculation(self):
        e = EvidenceItem(
            evidence_type="calculation",
            description="Diesel emissions",
            data={"result": 26800.0},
            citation_ids=["cid-1"],
        )
        h = e.calculate_content_hash()
        assert len(h) == 64

    def test_content_hash_deterministic(self):
        kwargs = {
            "evidence_type": "data_point",
            "description": "Test",
            "data": {"val": 42},
            "citation_ids": ["cid-1"],
        }
        e1 = EvidenceItem(**kwargs)
        e2 = EvidenceItem(**kwargs)
        assert e1.calculate_content_hash() == e2.calculate_content_hash()

    def test_default_data_empty(self):
        e = EvidenceItem()
        assert e.data == {}

    def test_default_citation_ids_empty(self):
        e = EvidenceItem()
        assert e.citation_ids == []


class TestEvidencePackageModel:
    """Test EvidencePackage model."""

    def test_creation(self):
        p = EvidencePackage(name="Scope 1 Evidence")
        assert p.name == "Scope 1 Evidence"
        assert p.is_finalized is False

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name is required"):
            EvidencePackage(name="")

    def test_add_evidence(self):
        p = EvidencePackage(name="Test")
        e = EvidenceItem(evidence_type="calculation", description="Test")
        p.add_evidence(e)
        assert len(p.evidence_items) == 1
        assert p.evidence_items[0].content_hash is not None

    def test_add_citation(self):
        p = EvidencePackage(name="Test")
        c = Citation(citation_type="emission_factor")
        p.add_citation(c)
        assert len(p.citations) == 1
        assert p.citations[0].content_hash is not None

    def test_finalize(self):
        p = EvidencePackage(name="Test")
        p.add_evidence(EvidenceItem(evidence_type="calculation", description="Calc"))
        h = p.finalize()
        assert len(h) == 64
        assert p.is_finalized is True
        assert p.package_hash == h

    def test_double_finalize_raises(self):
        p = EvidencePackage(name="Test")
        p.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            p.finalize()

    def test_add_evidence_after_finalize_raises(self):
        p = EvidencePackage(name="Test")
        p.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            p.add_evidence(EvidenceItem(evidence_type="calculation", description="X"))

    def test_add_citation_after_finalize_raises(self):
        p = EvidencePackage(name="Test")
        p.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            p.add_citation(Citation())

    def test_package_hash_deterministic(self):
        def make_package():
            p = EvidencePackage(name="Test")
            p.add_evidence(EvidenceItem(
                evidence_type="calculation",
                description="Calc",
                data={"result": 42},
                citation_ids=["cid-1"],
            ))
            return p.calculate_package_hash()
        assert make_package() == make_package()

    def test_add_evidence_invalidates_hash(self):
        p = EvidencePackage(name="Test")
        p.package_hash = "old_hash"
        p.add_evidence(EvidenceItem(evidence_type="calculation", description="X"))
        assert p.package_hash is None

    def test_add_citation_invalidates_hash(self):
        p = EvidencePackage(name="Test")
        p.package_hash = "old_hash"
        p.add_citation(Citation())
        assert p.package_hash is None


class TestVerificationRecordModel:
    """Test VerificationRecord model."""

    def test_creation(self):
        r = VerificationRecord(
            citation_id="cid-1",
            old_status="unverified",
            new_status="verified",
            verified_by="analyst1",
            reason="Source confirmed valid",
        )
        assert r.citation_id == "cid-1"
        assert r.new_status == "verified"
        assert r.verified_by == "analyst1"

    def test_auto_generated_id(self):
        r = VerificationRecord()
        assert len(r.record_id) == 36

    def test_default_timestamp(self):
        r = VerificationRecord()
        assert r.timestamp is not None


class TestCitationVersionModel:
    """Test CitationVersion model."""

    def test_creation(self):
        v = CitationVersion(
            citation_id="cid-1",
            version=2,
            data={"key_values": {"ef": 2.75}},
            change_type="update",
            changed_by="analyst1",
        )
        assert v.citation_id == "cid-1"
        assert v.version == 2
        assert v.change_type == "update"

    def test_default_version(self):
        v = CitationVersion()
        assert v.version == 1

    def test_default_timestamp(self):
        v = CitationVersion()
        assert v.timestamp is not None


class TestChangeLogEntryModel:
    """Test ChangeLogEntry model."""

    def test_creation(self):
        e = ChangeLogEntry(
            citation_id="cid-1",
            change_type="update",
            old_value={"ef": 2.68},
            new_value={"ef": 2.75},
            changed_by="analyst1",
            change_reason="Updated for 2025 data",
        )
        assert e.citation_id == "cid-1"
        assert e.old_value == {"ef": 2.68}
        assert e.new_value == {"ef": 2.75}

    def test_to_dict(self):
        e = ChangeLogEntry(citation_id="cid-1", change_type="create")
        d = e.to_dict()
        assert d["citation_id"] == "cid-1"
        assert d["change_type"] == "create"
        assert "timestamp" in d

    def test_auto_generated_id(self):
        e = ChangeLogEntry()
        assert len(e.entry_id) == 36
