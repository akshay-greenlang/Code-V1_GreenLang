# -*- coding: utf-8 -*-
"""
GL-FOUND-X-005: GreenLang Citations & Evidence SDK
====================================================

This package provides the citations management, evidence packaging,
verification, and provenance tracking SDK for the GreenLang framework.
It supports:

- Citation Registry: Store and retrieve citations with rich metadata
- Evidence Packaging: Bundle evidence with calculations for audit trails
- Source Verification: Verify citation sources are valid and current
- Provenance Tracking: SHA-256 chain hashing for tamper-evident audit trails
- Export/Import: BibTeX, JSON, and CSL-JSON format support
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_CITATIONS_ env prefix

Key Components:
    - registry: CitationRegistry for citation CRUD and versioning
    - evidence: EvidenceManager for evidence packaging
    - verification: VerificationEngine for source verification
    - provenance: ProvenanceTracker for SHA-256 audit trails
    - export_import: ExportImportManager for multi-format I/O
    - config: CitationsConfig with GL_CITATIONS_ env prefix
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: CitationsService facade

Example:
    >>> from greenlang.citations import CitationRegistry
    >>> r = CitationRegistry()
    >>> c = r.create(
    ...     citation_type="emission_factor",
    ...     source_authority="defra",
    ...     metadata={"title": "DEFRA GHG Conversion Factors 2024"},
    ...     effective_date="2024-01-01",
    ...     user_id="analyst",
    ...     change_reason="Initial registration",
    ... )
    >>> print(r.count)  # 1

    >>> from greenlang.citations import EvidenceManager
    >>> em = EvidenceManager()
    >>> pkg = em.create_package("Scope 1 Evidence Q4")
    >>> print(em.count)  # 1

Agent ID: GL-FOUND-X-005
Agent Name: Citations & Evidence
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-005"
__agent_name__ = "Citations & Evidence"

# SDK availability flag
CITATIONS_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.citations.config import (
    CitationsConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, core, verification, versioning, audit)
# ---------------------------------------------------------------------------
from greenlang.citations.models import (
    # Enumerations
    CitationType,
    SourceAuthority,
    RegulatoryFramework,
    VerificationStatus,
    EvidenceType,
    ExportFormat,
    ChangeType,
    # Core models
    CitationMetadata,
    Citation,
    EvidenceItem,
    EvidencePackage,
    MethodologyReference,
    RegulatoryRequirement,
    DataSourceAttribution,
    # Verification models
    VerificationRecord,
    # Versioning models
    CitationVersion,
    # Audit models
    ChangeLogEntry,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.citations.registry import CitationRegistry
from greenlang.citations.evidence import EvidenceManager
from greenlang.citations.verification import VerificationEngine
from greenlang.citations.provenance import ProvenanceTracker, ProvenanceEntry
from greenlang.citations.export_import import ExportImportManager

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.citations.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    citations_operations_total,
    citations_operation_duration_seconds,
    citations_verifications_total,
    citations_verification_failures_total,
    citations_evidence_packages_total,
    citations_evidence_items_total,
    citations_exports_total,
    citations_total,
    citations_packages_total,
    citations_cache_hits_total,
    citations_cache_misses_total,
    citations_provenance_chain_depth,
    # Helper functions
    record_operation,
    record_verification,
    record_verification_failure,
    record_evidence_package,
    record_evidence_item,
    record_export,
    update_citations_count,
    update_packages_count,
    record_cache_hit,
    record_cache_miss,
    record_provenance_depth,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.citations.setup import (
    CitationsService,
    configure_citations_service,
    get_citations_service,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "CITATIONS_SDK_AVAILABLE",
    # Configuration
    "CitationsConfig",
    "get_config",
    "set_config",
    "reset_config",
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
    # Verification models
    "VerificationRecord",
    # Versioning models
    "CitationVersion",
    # Audit models
    "ChangeLogEntry",
    # Core engines
    "CitationRegistry",
    "EvidenceManager",
    "VerificationEngine",
    "ProvenanceTracker",
    "ProvenanceEntry",
    "ExportImportManager",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "citations_operations_total",
    "citations_operation_duration_seconds",
    "citations_verifications_total",
    "citations_verification_failures_total",
    "citations_evidence_packages_total",
    "citations_evidence_items_total",
    "citations_exports_total",
    "citations_total",
    "citations_packages_total",
    "citations_cache_hits_total",
    "citations_cache_misses_total",
    "citations_provenance_chain_depth",
    # Metric helper functions
    "record_operation",
    "record_verification",
    "record_verification_failure",
    "record_evidence_package",
    "record_evidence_item",
    "record_export",
    "update_citations_count",
    "update_packages_count",
    "record_cache_hit",
    "record_cache_miss",
    "record_provenance_depth",
    # Service setup facade
    "CitationsService",
    "configure_citations_service",
    "get_citations_service",
    "get_router",
]
