# -*- coding: utf-8 -*-
"""
GL-FOUND-X-005: Citations & Evidence Agent
==========================================

The Citations & Evidence Agent provides complete provenance tracking for
zero-hallucination compliance. It manages citation registries, evidence
packaging, and source verification for all calculations and data.

Capabilities:
    - Citation Registry: Store and retrieve citations with rich metadata
    - Evidence Packaging: Package evidence with calculations for audit trails
    - Source Verification: Verify citation sources are valid and current
    - Methodology References: Link to GHG Protocol, ISO 14064, etc.
    - Regulatory Citations: Track CSRD, CBAM, EUDR, SB253 requirements
    - Data Source Attribution: Track DEFRA, EPA, Ecoinvent, IPCC sources
    - Citation Versioning: Handle annual updates to emission factors
    - Multiple Formats: Support BibTeX, JSON, and internal formats

Zero-Hallucination Principle:
    Every calculation must be traceable to a verified, citable source.
    No numeric value shall be generated without documented provenance.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock
from greenlang.utilities.determinism.uuid import deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class CitationType(str, Enum):
    """Types of citations supported by the system."""
    EMISSION_FACTOR = "emission_factor"      # DEFRA, EPA, Ecoinvent, IPCC
    REGULATORY = "regulatory"                 # EU, US, UK regulations
    METHODOLOGY = "methodology"               # GHG Protocol, ISO standards
    SCIENTIFIC = "scientific"                 # Peer-reviewed papers (DOI)
    COMPANY_DATA = "company_data"             # Internal data with timestamps
    GUIDANCE = "guidance"                     # Technical guidance documents
    DATABASE = "database"                     # Factor databases


class SourceAuthority(str, Enum):
    """Recognized data source authorities."""
    # Emission Factor Sources
    DEFRA = "defra"                           # UK DEFRA GHG Conversion Factors
    EPA = "epa"                               # US EPA Emission Factors
    ECOINVENT = "ecoinvent"                   # Ecoinvent LCA Database
    IPCC = "ipcc"                             # IPCC Guidelines & Factors
    GHG_PROTOCOL = "ghg_protocol"             # GHG Protocol Corporate Standard
    EXIOBASE = "exiobase"                     # EXIOBASE MRIO Database
    CLIMATIQ = "climatiq"                     # Climatiq API

    # Regulatory Bodies
    EU_COMMISSION = "eu_commission"           # European Commission
    SEC = "sec"                               # US SEC
    EFRAG = "efrag"                           # EFRAG (ESRS)
    CARB = "carb"                             # California ARB

    # Standards Bodies
    ISO = "iso"                               # ISO Standards
    GRI = "gri"                               # GRI Standards
    SASB = "sasb"                             # SASB Standards
    CDP = "cdp"                               # CDP Questionnaire

    # Internal
    INTERNAL = "internal"                     # Company internal data
    SUPPLIER = "supplier"                     # Supplier-provided data

    # Other
    OTHER = "other"                           # Other verified sources


class RegulatoryFramework(str, Enum):
    """Regulatory frameworks for compliance tracking."""
    CSRD = "csrd"                             # Corporate Sustainability Reporting Directive
    CBAM = "cbam"                             # Carbon Border Adjustment Mechanism
    EUDR = "eudr"                             # EU Deforestation Regulation
    SB253 = "sb253"                           # California Climate Accountability Act
    SB261 = "sb261"                           # California Climate-Related Risk Disclosure
    SEC_CLIMATE = "sec_climate"               # SEC Climate Disclosure Rules
    TCFD = "tcfd"                             # Task Force on Climate-related Financial Disclosures
    TNFD = "tnfd"                             # Taskforce on Nature-related Financial Disclosures


class VerificationStatus(str, Enum):
    """Status of citation verification."""
    VERIFIED = "verified"                     # Source verified and current
    PENDING = "pending"                       # Verification in progress
    EXPIRED = "expired"                       # Source has expired
    SUPERSEDED = "superseded"                 # Newer version available
    UNVERIFIED = "unverified"                 # Not yet verified
    INVALID = "invalid"                       # Source could not be verified


class EvidenceType(str, Enum):
    """Types of evidence that can be packaged."""
    CALCULATION = "calculation"               # Calculation with inputs/outputs
    DATA_POINT = "data_point"                 # Single data value
    METHODOLOGY = "methodology"               # Methodology reference
    ASSUMPTION = "assumption"                 # Documented assumption
    VALIDATION = "validation"                 # Validation result
    AUDIT_TRAIL = "audit_trail"               # Complete audit trail


# =============================================================================
# Pydantic Models
# =============================================================================

class CitationMetadata(BaseModel):
    """Metadata for a citation."""

    title: str = Field(..., description="Title of the cited work")
    authors: List[str] = Field(default_factory=list, description="Authors of the work")
    publication_date: Optional[date] = Field(None, description="Publication date")
    version: Optional[str] = Field(None, description="Version number (e.g., '2024')")
    edition: Optional[str] = Field(None, description="Edition if applicable")
    publisher: Optional[str] = Field(None, description="Publisher or organization")
    url: Optional[str] = Field(None, description="URL for online access")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    isbn: Optional[str] = Field(None, description="ISBN for books")
    issn: Optional[str] = Field(None, description="ISSN for journals")
    page_numbers: Optional[str] = Field(None, description="Page range (e.g., '15-23')")
    chapter: Optional[str] = Field(None, description="Chapter name or number")
    section: Optional[str] = Field(None, description="Section reference")
    table_reference: Optional[str] = Field(None, description="Table number/name")

    @field_validator('doi')
    @classmethod
    def validate_doi(cls, v: Optional[str]) -> Optional[str]:
        """Validate DOI format."""
        if v is None:
            return v
        # DOI format: 10.xxxx/xxxxx
        if not re.match(r'^10\.\d{4,}/[^\s]+$', v):
            raise ValueError(f"Invalid DOI format: {v}")
        return v


class Citation(BaseModel):
    """
    Complete citation record for zero-hallucination compliance.

    Every data point used in calculations must be traceable to a Citation.
    """

    citation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this citation"
    )
    citation_type: CitationType = Field(
        ...,
        description="Type of citation"
    )
    source_authority: SourceAuthority = Field(
        ...,
        description="Authority/organization providing the data"
    )
    metadata: CitationMetadata = Field(
        ...,
        description="Citation metadata"
    )

    # Versioning
    effective_date: date = Field(
        ...,
        description="Date from which this citation is effective"
    )
    expiration_date: Optional[date] = Field(
        None,
        description="Date when this citation expires"
    )
    superseded_by: Optional[str] = Field(
        None,
        description="Citation ID that supersedes this one"
    )
    supersedes: Optional[str] = Field(
        None,
        description="Citation ID that this one supersedes"
    )

    # Verification
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Verification status"
    )
    verified_at: Optional[datetime] = Field(
        None,
        description="When verification was performed"
    )
    verified_by: Optional[str] = Field(
        None,
        description="Who verified this citation"
    )

    # Regulatory linkage
    regulatory_frameworks: List[RegulatoryFramework] = Field(
        default_factory=list,
        description="Regulatory frameworks this citation supports"
    )

    # Content
    abstract: Optional[str] = Field(
        None,
        description="Brief description of the content"
    )
    key_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key values/factors from this citation"
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes"
    )

    # Audit trail
    created_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When this record was created"
    )
    updated_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When this record was last updated"
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created this record"
    )

    # Provenance hash
    content_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of citation content"
    )

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of citation content for provenance."""
        content = {
            "citation_type": self.citation_type.value,
            "source_authority": self.source_authority.value,
            "metadata": self.metadata.model_dump(),
            "effective_date": self.effective_date.isoformat(),
            "key_values": self.key_values,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def is_valid(self, reference_date: Optional[date] = None) -> bool:
        """Check if citation is valid as of a given date."""
        if reference_date is None:
            reference_date = date.today()

        # Check effective date
        if self.effective_date > reference_date:
            return False

        # Check expiration
        if self.expiration_date and self.expiration_date < reference_date:
            return False

        # Check verification status
        if self.verification_status in (
            VerificationStatus.INVALID,
            VerificationStatus.EXPIRED
        ):
            return False

        return True

    def to_bibtex(self) -> str:
        """Export citation to BibTeX format."""
        entry_type = self._get_bibtex_type()
        bibtex_id = self._generate_bibtex_id()

        fields = []

        if self.metadata.title:
            fields.append(f'  title = {{{self.metadata.title}}}')

        if self.metadata.authors:
            authors_str = ' and '.join(self.metadata.authors)
            fields.append(f'  author = {{{authors_str}}}')

        if self.metadata.publication_date:
            fields.append(f'  year = {{{self.metadata.publication_date.year}}}')

        if self.metadata.publisher:
            fields.append(f'  publisher = {{{self.metadata.publisher}}}')

        if self.metadata.url:
            fields.append(f'  url = {{{self.metadata.url}}}')

        if self.metadata.doi:
            fields.append(f'  doi = {{{self.metadata.doi}}}')

        if self.metadata.isbn:
            fields.append(f'  isbn = {{{self.metadata.isbn}}}')

        if self.metadata.version:
            fields.append(f'  edition = {{{self.metadata.version}}}')

        fields_str = ',\n'.join(fields)
        return f'@{entry_type}{{{bibtex_id},\n{fields_str}\n}}'

    def _get_bibtex_type(self) -> str:
        """Determine BibTeX entry type."""
        type_mapping = {
            CitationType.EMISSION_FACTOR: 'techreport',
            CitationType.REGULATORY: 'misc',
            CitationType.METHODOLOGY: 'manual',
            CitationType.SCIENTIFIC: 'article',
            CitationType.COMPANY_DATA: 'misc',
            CitationType.GUIDANCE: 'techreport',
            CitationType.DATABASE: 'misc',
        }
        return type_mapping.get(self.citation_type, 'misc')

    def _generate_bibtex_id(self) -> str:
        """Generate a BibTeX-friendly ID."""
        # Use first author surname + year
        if self.metadata.authors:
            surname = self.metadata.authors[0].split()[-1].lower()
        else:
            surname = self.source_authority.value

        year = ""
        if self.metadata.publication_date:
            year = str(self.metadata.publication_date.year)

        # Clean the ID
        bibtex_id = re.sub(r'[^a-z0-9]', '', f"{surname}{year}")
        return bibtex_id or self.citation_id[:8]


class EvidenceItem(BaseModel):
    """A single piece of evidence in an evidence package."""

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evidence"
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence"
    )
    description: str = Field(
        ...,
        description="Description of what this evidence demonstrates"
    )

    # Data
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The evidence data"
    )

    # Citations
    citation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of citations supporting this evidence"
    )

    # Provenance
    source_system: Optional[str] = Field(
        None,
        description="System that produced this evidence"
    )
    source_agent: Optional[str] = Field(
        None,
        description="Agent that produced this evidence"
    )
    timestamp: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When this evidence was recorded"
    )

    # Hash for integrity
    content_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of evidence content"
    )

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
    """
    Complete evidence package for audit-ready documentation.

    Bundles all evidence, citations, and calculations for a specific
    calculation or data point.
    """

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this package"
    )
    name: str = Field(
        ...,
        description="Name of this evidence package"
    )
    description: str = Field(
        default="",
        description="Description of what this package documents"
    )

    # Content
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items in this package"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations included in this package"
    )

    # Context
    calculation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context for calculations (inputs, parameters)"
    )
    calculation_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results of calculations"
    )

    # Regulatory
    regulatory_frameworks: List[RegulatoryFramework] = Field(
        default_factory=list,
        description="Regulatory frameworks this package supports"
    )
    compliance_notes: Optional[str] = Field(
        None,
        description="Notes on regulatory compliance"
    )

    # Audit metadata
    created_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When this package was created"
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created this package"
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant identifier"
    )

    # Integrity
    package_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of entire package"
    )

    def calculate_package_hash(self) -> str:
        """Calculate SHA-256 hash of entire package."""
        content = {
            "name": self.name,
            "evidence_items": [
                item.calculate_content_hash()
                for item in self.evidence_items
            ],
            "citations": [
                citation.calculate_content_hash()
                for citation in self.citations
            ],
            "calculation_context": self.calculation_context,
            "calculation_result": self.calculation_result,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def add_evidence(self, evidence: EvidenceItem) -> None:
        """Add an evidence item to the package."""
        evidence.content_hash = evidence.calculate_content_hash()
        self.evidence_items.append(evidence)
        self.package_hash = None  # Invalidate package hash

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the package."""
        citation.content_hash = citation.calculate_content_hash()
        self.citations.append(citation)
        self.package_hash = None  # Invalidate package hash

    def finalize(self) -> str:
        """Finalize the package and return its hash."""
        self.package_hash = self.calculate_package_hash()
        return self.package_hash

    def to_json(self, indent: int = 2) -> str:
        """Export package to JSON format."""
        return self.model_dump_json(indent=indent)


class MethodologyReference(BaseModel):
    """Reference to a calculation methodology."""

    reference_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    name: str = Field(
        ...,
        description="Name of the methodology"
    )
    standard: str = Field(
        ...,
        description="Standard (e.g., 'GHG Protocol', 'ISO 14064-1')"
    )
    version: str = Field(
        ...,
        description="Version of the standard"
    )
    section: Optional[str] = Field(
        None,
        description="Specific section reference"
    )
    description: str = Field(
        default="",
        description="Description of the methodology"
    )

    # Associated citation
    citation_id: Optional[str] = Field(
        None,
        description="ID of the supporting citation"
    )

    # Applicability
    scope_1_applicable: bool = Field(default=False)
    scope_2_applicable: bool = Field(default=False)
    scope_3_applicable: bool = Field(default=False)
    applicable_categories: List[str] = Field(
        default_factory=list,
        description="Applicable Scope 3 categories"
    )

    # Formula reference
    formula_id: Optional[str] = Field(
        None,
        description="ID of the formula in the formula registry"
    )


class RegulatoryRequirement(BaseModel):
    """A regulatory requirement for compliance tracking."""

    requirement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    framework: RegulatoryFramework = Field(
        ...,
        description="Regulatory framework"
    )
    article: Optional[str] = Field(
        None,
        description="Article number (e.g., 'Article 29b')"
    )
    requirement_text: str = Field(
        ...,
        description="Text of the requirement"
    )

    # Citation
    citation_id: Optional[str] = Field(
        None,
        description="ID of the source citation"
    )

    # Dates
    effective_date: date = Field(
        ...,
        description="When requirement becomes effective"
    )
    compliance_deadline: Optional[date] = Field(
        None,
        description="Deadline for compliance"
    )

    # Scope
    applies_to_scope_1: bool = Field(default=False)
    applies_to_scope_2: bool = Field(default=False)
    applies_to_scope_3: bool = Field(default=False)

    # Compliance
    compliance_status: Optional[str] = Field(
        None,
        description="Current compliance status"
    )
    compliance_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence package IDs demonstrating compliance"
    )


class DataSourceAttribution(BaseModel):
    """Attribution for a data source."""

    attribution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    source_authority: SourceAuthority = Field(
        ...,
        description="Source authority"
    )
    dataset_name: str = Field(
        ...,
        description="Name of the dataset"
    )
    dataset_version: str = Field(
        ...,
        description="Version of the dataset"
    )

    # Citation
    citation_id: Optional[str] = Field(
        None,
        description="ID of the source citation"
    )

    # Data extracted
    extracted_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values extracted from this source"
    )
    extraction_date: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When data was extracted"
    )
    extracted_by: Optional[str] = Field(
        None,
        description="Who extracted the data"
    )

    # Validity
    valid_from: date = Field(
        ...,
        description="Start of validity period"
    )
    valid_until: Optional[date] = Field(
        None,
        description="End of validity period"
    )


# =============================================================================
# Agent Input/Output Models
# =============================================================================

class CitationsAgentInput(BaseModel):
    """Input for Citations & Evidence Agent."""

    action: str = Field(
        ...,
        description="Action to perform: register, lookup, verify, package, export"
    )

    # For registration
    citation: Optional[Citation] = Field(
        None,
        description="Citation to register"
    )
    evidence: Optional[EvidenceItem] = Field(
        None,
        description="Evidence item to add"
    )

    # For lookup
    citation_id: Optional[str] = Field(
        None,
        description="Citation ID to look up"
    )
    citation_ids: Optional[List[str]] = Field(
        None,
        description="Multiple citation IDs to look up"
    )

    # For queries
    query_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters for querying citations"
    )

    # For packaging
    package_name: Optional[str] = Field(
        None,
        description="Name for evidence package"
    )
    calculation_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Calculation context"
    )
    calculation_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Calculation result"
    )

    # For export
    export_format: Optional[str] = Field(
        None,
        description="Export format: bibtex, json"
    )

    # Metadata
    user_id: Optional[str] = Field(
        None,
        description="User performing the action"
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant identifier"
    )

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is supported."""
        valid_actions = {
            'register_citation',
            'lookup_citation',
            'lookup_multiple',
            'verify_citation',
            'verify_sources',
            'create_package',
            'add_evidence',
            'finalize_package',
            'export_citations',
            'query_citations',
            'get_methodology',
            'get_regulatory',
            'check_validity',
        }
        if v not in valid_actions:
            raise ValueError(f"Invalid action: {v}. Valid: {valid_actions}")
        return v


class CitationsAgentOutput(BaseModel):
    """Output from Citations & Evidence Agent."""

    success: bool = Field(
        ...,
        description="Whether the operation succeeded"
    )
    action: str = Field(
        ...,
        description="Action that was performed"
    )

    # Results
    citation: Optional[Citation] = Field(
        None,
        description="Single citation result"
    )
    citations: Optional[List[Citation]] = Field(
        None,
        description="Multiple citation results"
    )
    evidence_package: Optional[EvidencePackage] = Field(
        None,
        description="Evidence package result"
    )

    # Verification results
    verification_results: Optional[Dict[str, VerificationStatus]] = Field(
        None,
        description="Verification status by citation ID"
    )
    validity_results: Optional[Dict[str, bool]] = Field(
        None,
        description="Validity status by citation ID"
    )

    # Export
    exported_content: Optional[str] = Field(
        None,
        description="Exported content (BibTeX, JSON)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the result"
    )

    # Error handling
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When this result was generated"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )


# =============================================================================
# Citations & Evidence Agent
# =============================================================================

class CitationsEvidenceAgent(BaseAgent):
    """
    GL-FOUND-X-005: Citations & Evidence Agent

    Provides complete provenance tracking for zero-hallucination compliance.
    Manages citation registries, evidence packaging, and source verification.

    Zero-Hallucination Guarantees:
        - Every calculation has traceable citations
        - All sources are verified and versioned
        - Complete audit trail with SHA-256 hashes
        - Evidence packages are tamper-evident

    Usage:
        agent = CitationsEvidenceAgent()
        result = agent.run({
            'action': 'register_citation',
            'citation': citation_data
        })
    """

    AGENT_ID = "GL-FOUND-X-005"
    AGENT_NAME = "Citations & Evidence Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Citations & Evidence Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Citation registry and evidence packaging for zero-hallucination compliance",
                version=self.VERSION,
                parameters={
                    "enable_auto_verification": True,
                    "default_expiration_years": 5,
                    "enable_hash_validation": True,
                }
            )
        super().__init__(config)

        # Citation registry (in-memory, replace with database in production)
        self._citations: Dict[str, Citation] = {}

        # Evidence packages
        self._packages: Dict[str, EvidencePackage] = {}

        # Methodology references
        self._methodologies: Dict[str, MethodologyReference] = {}

        # Regulatory requirements
        self._requirements: Dict[str, RegulatoryRequirement] = {}

        # Data source attributions
        self._attributions: Dict[str, DataSourceAttribution] = {}

        # Initialize with standard methodologies
        self._initialize_standard_methodologies()

        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def _initialize_standard_methodologies(self) -> None:
        """Initialize standard methodology references."""
        # GHG Protocol Corporate Standard
        ghg_corporate = MethodologyReference(
            reference_id="ghg-protocol-corporate",
            name="GHG Protocol Corporate Standard",
            standard="GHG Protocol",
            version="Revised Edition",
            description="Corporate Accounting and Reporting Standard",
            scope_1_applicable=True,
            scope_2_applicable=True,
            scope_3_applicable=False,
        )
        self._methodologies[ghg_corporate.reference_id] = ghg_corporate

        # GHG Protocol Scope 3 Standard
        ghg_scope3 = MethodologyReference(
            reference_id="ghg-protocol-scope3",
            name="GHG Protocol Corporate Value Chain (Scope 3) Standard",
            standard="GHG Protocol",
            version="2011",
            description="Scope 3 Accounting and Reporting Standard",
            scope_3_applicable=True,
            applicable_categories=[
                "Category 1: Purchased Goods and Services",
                "Category 2: Capital Goods",
                "Category 3: Fuel and Energy-Related Activities",
                "Category 4: Upstream Transportation and Distribution",
                "Category 5: Waste Generated in Operations",
                "Category 6: Business Travel",
                "Category 7: Employee Commuting",
                "Category 8: Upstream Leased Assets",
                "Category 9: Downstream Transportation and Distribution",
                "Category 10: Processing of Sold Products",
                "Category 11: Use of Sold Products",
                "Category 12: End-of-Life Treatment of Sold Products",
                "Category 13: Downstream Leased Assets",
                "Category 14: Franchises",
                "Category 15: Investments",
            ],
        )
        self._methodologies[ghg_scope3.reference_id] = ghg_scope3

        # ISO 14064-1
        iso_14064 = MethodologyReference(
            reference_id="iso-14064-1",
            name="ISO 14064-1:2018",
            standard="ISO 14064-1",
            version="2018",
            description="Specification with guidance for quantification and reporting of GHG emissions",
            scope_1_applicable=True,
            scope_2_applicable=True,
            scope_3_applicable=True,
        )
        self._methodologies[iso_14064.reference_id] = iso_14064

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute citation/evidence operation.

        Args:
            input_data: Input parameters including action and data

        Returns:
            AgentResult containing operation results
        """
        import time
        start_time = time.time()

        try:
            # Parse and validate input
            agent_input = CitationsAgentInput(**input_data)

            # Route to appropriate handler
            action_handlers = {
                'register_citation': self._handle_register_citation,
                'lookup_citation': self._handle_lookup_citation,
                'lookup_multiple': self._handle_lookup_multiple,
                'verify_citation': self._handle_verify_citation,
                'verify_sources': self._handle_verify_sources,
                'create_package': self._handle_create_package,
                'add_evidence': self._handle_add_evidence,
                'finalize_package': self._handle_finalize_package,
                'export_citations': self._handle_export_citations,
                'query_citations': self._handle_query_citations,
                'get_methodology': self._handle_get_methodology,
                'get_regulatory': self._handle_get_regulatory,
                'check_validity': self._handle_check_validity,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)

            # Calculate processing time
            output.processing_time_ms = (time.time() - start_time) * 1000

            # Calculate provenance hash
            output.provenance_hash = self._calculate_output_hash(output)

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                error=output.error,
            )

        except Exception as e:
            self.logger.error(f"Citation operation failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
            )

    def _handle_register_citation(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Register a new citation."""
        if not input_data.citation:
            return CitationsAgentOutput(
                success=False,
                action='register_citation',
                error="No citation provided"
            )

        citation = input_data.citation

        # Calculate content hash
        citation.content_hash = citation.calculate_content_hash()

        # Auto-verify if enabled
        if self.config.parameters.get('enable_auto_verification'):
            citation.verification_status = VerificationStatus.PENDING

        # Store citation
        self._citations[citation.citation_id] = citation

        self.logger.info(f"Registered citation: {citation.citation_id}")

        return CitationsAgentOutput(
            success=True,
            action='register_citation',
            citation=citation,
        )

    def _handle_lookup_citation(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Look up a single citation by ID."""
        if not input_data.citation_id:
            return CitationsAgentOutput(
                success=False,
                action='lookup_citation',
                error="No citation_id provided"
            )

        citation = self._citations.get(input_data.citation_id)

        if not citation:
            return CitationsAgentOutput(
                success=False,
                action='lookup_citation',
                error=f"Citation not found: {input_data.citation_id}"
            )

        return CitationsAgentOutput(
            success=True,
            action='lookup_citation',
            citation=citation,
        )

    def _handle_lookup_multiple(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Look up multiple citations by IDs."""
        if not input_data.citation_ids:
            return CitationsAgentOutput(
                success=False,
                action='lookup_multiple',
                error="No citation_ids provided"
            )

        citations = []
        warnings = []

        for cid in input_data.citation_ids:
            citation = self._citations.get(cid)
            if citation:
                citations.append(citation)
            else:
                warnings.append(f"Citation not found: {cid}")

        return CitationsAgentOutput(
            success=True,
            action='lookup_multiple',
            citations=citations,
            warnings=warnings,
        )

    def _handle_verify_citation(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Verify a single citation."""
        if not input_data.citation_id:
            return CitationsAgentOutput(
                success=False,
                action='verify_citation',
                error="No citation_id provided"
            )

        citation = self._citations.get(input_data.citation_id)

        if not citation:
            return CitationsAgentOutput(
                success=False,
                action='verify_citation',
                error=f"Citation not found: {input_data.citation_id}"
            )

        # Perform verification
        verification_status = self._verify_single_citation(citation)

        # Update citation
        citation.verification_status = verification_status
        citation.verified_at = DeterministicClock.now()
        citation.verified_by = input_data.user_id
        citation.updated_at = DeterministicClock.now()

        return CitationsAgentOutput(
            success=True,
            action='verify_citation',
            citation=citation,
            verification_results={citation.citation_id: verification_status},
        )

    def _handle_verify_sources(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Verify multiple citation sources."""
        citation_ids = input_data.citation_ids or list(self._citations.keys())

        results: Dict[str, VerificationStatus] = {}
        warnings = []

        for cid in citation_ids:
            citation = self._citations.get(cid)
            if citation:
                status = self._verify_single_citation(citation)
                citation.verification_status = status
                citation.verified_at = DeterministicClock.now()
                results[cid] = status
            else:
                warnings.append(f"Citation not found: {cid}")

        return CitationsAgentOutput(
            success=True,
            action='verify_sources',
            verification_results=results,
            warnings=warnings,
        )

    def _verify_single_citation(self, citation: Citation) -> VerificationStatus:
        """
        Verify a single citation.

        In production, this would:
        - Check if URL is accessible
        - Verify DOI resolves
        - Check if source has been superseded
        - Validate against known source registries
        """
        # Check expiration
        if citation.expiration_date and citation.expiration_date < date.today():
            return VerificationStatus.EXPIRED

        # Check if superseded
        if citation.superseded_by:
            return VerificationStatus.SUPERSEDED

        # Check content hash integrity
        if citation.content_hash:
            current_hash = citation.calculate_content_hash()
            if current_hash != citation.content_hash:
                self.logger.warning(
                    f"Content hash mismatch for citation {citation.citation_id}"
                )
                return VerificationStatus.INVALID

        # Validate required fields based on source authority
        if citation.source_authority in (
            SourceAuthority.DEFRA,
            SourceAuthority.EPA,
            SourceAuthority.ECOINVENT
        ):
            if not citation.metadata.version:
                return VerificationStatus.UNVERIFIED

        if citation.citation_type == CitationType.SCIENTIFIC:
            if not citation.metadata.doi:
                return VerificationStatus.UNVERIFIED

        return VerificationStatus.VERIFIED

    def _handle_create_package(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Create a new evidence package."""
        if not input_data.package_name:
            return CitationsAgentOutput(
                success=False,
                action='create_package',
                error="No package_name provided"
            )

        package = EvidencePackage(
            name=input_data.package_name,
            description=f"Evidence package for {input_data.package_name}",
            calculation_context=input_data.calculation_context or {},
            calculation_result=input_data.calculation_result or {},
            created_by=input_data.user_id,
            tenant_id=input_data.tenant_id,
        )

        # Add citations if provided
        if input_data.citation_ids:
            for cid in input_data.citation_ids:
                citation = self._citations.get(cid)
                if citation:
                    package.add_citation(citation)

        self._packages[package.package_id] = package

        self.logger.info(f"Created evidence package: {package.package_id}")

        return CitationsAgentOutput(
            success=True,
            action='create_package',
            evidence_package=package,
        )

    def _handle_add_evidence(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Add evidence to an existing package."""
        # This requires a package_id in query_filters
        package_id = (input_data.query_filters or {}).get('package_id')

        if not package_id:
            return CitationsAgentOutput(
                success=False,
                action='add_evidence',
                error="No package_id provided in query_filters"
            )

        if not input_data.evidence:
            return CitationsAgentOutput(
                success=False,
                action='add_evidence',
                error="No evidence provided"
            )

        package = self._packages.get(package_id)
        if not package:
            return CitationsAgentOutput(
                success=False,
                action='add_evidence',
                error=f"Package not found: {package_id}"
            )

        package.add_evidence(input_data.evidence)

        return CitationsAgentOutput(
            success=True,
            action='add_evidence',
            evidence_package=package,
        )

    def _handle_finalize_package(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Finalize an evidence package with hash."""
        package_id = (input_data.query_filters or {}).get('package_id')

        if not package_id:
            return CitationsAgentOutput(
                success=False,
                action='finalize_package',
                error="No package_id provided in query_filters"
            )

        package = self._packages.get(package_id)
        if not package:
            return CitationsAgentOutput(
                success=False,
                action='finalize_package',
                error=f"Package not found: {package_id}"
            )

        package_hash = package.finalize()

        self.logger.info(
            f"Finalized evidence package: {package_id} with hash {package_hash[:16]}..."
        )

        return CitationsAgentOutput(
            success=True,
            action='finalize_package',
            evidence_package=package,
            provenance_hash=package_hash,
        )

    def _handle_export_citations(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Export citations to specified format."""
        export_format = input_data.export_format or 'json'
        citation_ids = input_data.citation_ids or list(self._citations.keys())

        citations = [
            self._citations[cid]
            for cid in citation_ids
            if cid in self._citations
        ]

        if export_format.lower() == 'bibtex':
            content = self._export_bibtex(citations)
        elif export_format.lower() == 'json':
            content = self._export_json(citations)
        else:
            return CitationsAgentOutput(
                success=False,
                action='export_citations',
                error=f"Unsupported export format: {export_format}"
            )

        return CitationsAgentOutput(
            success=True,
            action='export_citations',
            exported_content=content,
            citations=citations,
        )

    def _export_bibtex(self, citations: List[Citation]) -> str:
        """Export citations to BibTeX format."""
        entries = [citation.to_bibtex() for citation in citations]
        return '\n\n'.join(entries)

    def _export_json(self, citations: List[Citation]) -> str:
        """Export citations to JSON format."""
        data = [citation.model_dump() for citation in citations]
        return json.dumps(data, indent=2, default=str)

    def _handle_query_citations(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Query citations with filters."""
        filters = input_data.query_filters or {}

        results = list(self._citations.values())

        # Apply filters
        if 'citation_type' in filters:
            ct = CitationType(filters['citation_type'])
            results = [c for c in results if c.citation_type == ct]

        if 'source_authority' in filters:
            sa = SourceAuthority(filters['source_authority'])
            results = [c for c in results if c.source_authority == sa]

        if 'verification_status' in filters:
            vs = VerificationStatus(filters['verification_status'])
            results = [c for c in results if c.verification_status == vs]

        if 'regulatory_framework' in filters:
            rf = RegulatoryFramework(filters['regulatory_framework'])
            results = [c for c in results if rf in c.regulatory_frameworks]

        if 'effective_after' in filters:
            after_date = date.fromisoformat(filters['effective_after'])
            results = [c for c in results if c.effective_date >= after_date]

        if 'effective_before' in filters:
            before_date = date.fromisoformat(filters['effective_before'])
            results = [c for c in results if c.effective_date <= before_date]

        if 'valid_only' in filters and filters['valid_only']:
            results = [c for c in results if c.is_valid()]

        return CitationsAgentOutput(
            success=True,
            action='query_citations',
            citations=results,
        )

    def _handle_get_methodology(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Get methodology reference."""
        methodology_id = (input_data.query_filters or {}).get('methodology_id')

        if methodology_id:
            methodology = self._methodologies.get(methodology_id)
            if methodology:
                # Convert to citation-like format for output
                citations = [self._methodology_to_citation(methodology)]
                return CitationsAgentOutput(
                    success=True,
                    action='get_methodology',
                    citations=citations,
                )
            else:
                return CitationsAgentOutput(
                    success=False,
                    action='get_methodology',
                    error=f"Methodology not found: {methodology_id}"
                )
        else:
            # Return all methodologies
            citations = [
                self._methodology_to_citation(m)
                for m in self._methodologies.values()
            ]
            return CitationsAgentOutput(
                success=True,
                action='get_methodology',
                citations=citations,
            )

    def _methodology_to_citation(
        self,
        methodology: MethodologyReference
    ) -> Citation:
        """Convert methodology reference to citation format."""
        return Citation(
            citation_id=methodology.reference_id,
            citation_type=CitationType.METHODOLOGY,
            source_authority=SourceAuthority.GHG_PROTOCOL,
            metadata=CitationMetadata(
                title=methodology.name,
                version=methodology.version,
            ),
            effective_date=date(2015, 1, 1),  # Default for GHG Protocol
            verification_status=VerificationStatus.VERIFIED,
            abstract=methodology.description,
        )

    def _handle_get_regulatory(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Get regulatory requirements."""
        filters = input_data.query_filters or {}

        results = list(self._requirements.values())

        if 'framework' in filters:
            rf = RegulatoryFramework(filters['framework'])
            results = [r for r in results if r.framework == rf]

        # Convert to citation format
        citations = [self._requirement_to_citation(r) for r in results]

        return CitationsAgentOutput(
            success=True,
            action='get_regulatory',
            citations=citations,
        )

    def _requirement_to_citation(
        self,
        requirement: RegulatoryRequirement
    ) -> Citation:
        """Convert regulatory requirement to citation format."""
        return Citation(
            citation_id=requirement.requirement_id,
            citation_type=CitationType.REGULATORY,
            source_authority=SourceAuthority.EU_COMMISSION,
            metadata=CitationMetadata(
                title=f"{requirement.framework.value.upper()} - {requirement.article or 'General'}",
            ),
            effective_date=requirement.effective_date,
            regulatory_frameworks=[requirement.framework],
            abstract=requirement.requirement_text,
        )

    def _handle_check_validity(
        self,
        input_data: CitationsAgentInput
    ) -> CitationsAgentOutput:
        """Check validity of citations."""
        citation_ids = input_data.citation_ids or list(self._citations.keys())

        reference_date_str = (input_data.query_filters or {}).get('reference_date')
        reference_date = (
            date.fromisoformat(reference_date_str)
            if reference_date_str
            else None
        )

        validity_results: Dict[str, bool] = {}
        warnings = []

        for cid in citation_ids:
            citation = self._citations.get(cid)
            if citation:
                validity_results[cid] = citation.is_valid(reference_date)
                if not validity_results[cid]:
                    warnings.append(f"Citation {cid} is not valid")
            else:
                warnings.append(f"Citation not found: {cid}")

        return CitationsAgentOutput(
            success=True,
            action='check_validity',
            validity_results=validity_results,
            warnings=warnings,
        )

    def _calculate_output_hash(self, output: CitationsAgentOutput) -> str:
        """Calculate SHA-256 hash of output for provenance."""
        content = {
            "action": output.action,
            "success": output.success,
            "timestamp": output.timestamp.isoformat(),
        }

        if output.citation:
            content["citation_hash"] = output.citation.calculate_content_hash()

        if output.citations:
            content["citations_hash"] = [
                c.calculate_content_hash() for c in output.citations
            ]

        if output.evidence_package:
            content["package_hash"] = output.evidence_package.calculate_package_hash()

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    # =========================================================================
    # Public API Methods (for direct usage without going through execute)
    # =========================================================================

    def register_citation(self, citation: Citation) -> str:
        """
        Register a citation directly.

        Args:
            citation: Citation to register

        Returns:
            Citation ID
        """
        citation.content_hash = citation.calculate_content_hash()
        self._citations[citation.citation_id] = citation
        return citation.citation_id

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by ID."""
        return self._citations.get(citation_id)

    def get_all_citations(self) -> List[Citation]:
        """Get all registered citations."""
        return list(self._citations.values())

    def create_evidence_package(
        self,
        name: str,
        calculation_context: Dict[str, Any],
        calculation_result: Dict[str, Any],
        citation_ids: List[str],
        user_id: Optional[str] = None,
    ) -> EvidencePackage:
        """
        Create a complete evidence package.

        Args:
            name: Package name
            calculation_context: Input context for calculation
            calculation_result: Output of calculation
            citation_ids: IDs of supporting citations
            user_id: User creating the package

        Returns:
            Finalized evidence package
        """
        package = EvidencePackage(
            name=name,
            calculation_context=calculation_context,
            calculation_result=calculation_result,
            created_by=user_id,
        )

        # Add citations
        for cid in citation_ids:
            citation = self._citations.get(cid)
            if citation:
                package.add_citation(citation)

        # Add calculation evidence
        calc_evidence = EvidenceItem(
            evidence_type=EvidenceType.CALCULATION,
            description=f"Calculation for {name}",
            data={
                "inputs": calculation_context,
                "outputs": calculation_result,
            },
            citation_ids=citation_ids,
        )
        package.add_evidence(calc_evidence)

        # Finalize
        package.finalize()

        # Store
        self._packages[package.package_id] = package

        return package

    def get_evidence_package(self, package_id: str) -> Optional[EvidencePackage]:
        """Get an evidence package by ID."""
        return self._packages.get(package_id)

    def register_regulatory_requirement(
        self,
        requirement: RegulatoryRequirement
    ) -> str:
        """Register a regulatory requirement."""
        self._requirements[requirement.requirement_id] = requirement
        return requirement.requirement_id

    def get_citations_for_framework(
        self,
        framework: RegulatoryFramework
    ) -> List[Citation]:
        """Get all citations relevant to a regulatory framework."""
        return [
            c for c in self._citations.values()
            if framework in c.regulatory_frameworks
        ]

    def get_valid_citations(
        self,
        reference_date: Optional[date] = None
    ) -> List[Citation]:
        """Get all currently valid citations."""
        return [
            c for c in self._citations.values()
            if c.is_valid(reference_date)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        base_stats = super().get_stats()

        # Add citation-specific stats
        base_stats.update({
            "total_citations": len(self._citations),
            "citations_by_type": self._count_by_field(
                self._citations.values(),
                lambda c: c.citation_type.value
            ),
            "citations_by_status": self._count_by_field(
                self._citations.values(),
                lambda c: c.verification_status.value
            ),
            "total_packages": len(self._packages),
            "total_methodologies": len(self._methodologies),
            "total_requirements": len(self._requirements),
        })

        return base_stats

    def _count_by_field(
        self,
        items: Any,
        field_extractor: Any
    ) -> Dict[str, int]:
        """Count items by a field value."""
        counts: Dict[str, int] = {}
        for item in items:
            key = field_extractor(item)
            counts[key] = counts.get(key, 0) + 1
        return counts
