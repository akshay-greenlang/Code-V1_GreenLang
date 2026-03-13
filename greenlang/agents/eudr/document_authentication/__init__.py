# -*- coding: utf-8 -*-
"""
Document Authentication Agent - AGENT-EUDR-012

Production-grade document authentication engine for EUDR compliance
covering document classification, digital signature verification,
hash integrity validation, certificate chain validation, metadata
extraction, fraud pattern detection, cross-reference verification
against external registries, and compliance reporting with evidence
packages.

This package provides a comprehensive document authentication system
for EUDR supply chain traceability supporting document integrity
verification per EU 2023/1115 Articles 4, 10, and 14:

    Capabilities:
        - Document classification with confidence scoring across 20
          document types (COO, phytosanitary certs, BOL, FSC/RSPO/
          ISCC/Fairtrade/UTZ certificates, due diligence statements,
          invoices, timber concessions, waste quality certificates)
        - Digital signature verification for CAdES, PAdES, XAdES,
          JAdES, QES, PGP, and PKCS#7 standards with timestamp
          validation and self-signed certificate handling
        - Hash integrity validation with SHA-256 primary and SHA-512
          secondary algorithms, five-year registry TTL, and duplicate
          document detection via content fingerprinting
        - Certificate chain validation with OCSP stapling, CRL
          refresh, key size enforcement (RSA 2048+, ECDSA 256+),
          and optional certificate transparency log checking
        - Metadata extraction and anomaly detection for creation date
          tolerance, author matching, GPS coordinate extraction,
          and empty metadata flagging
        - Fraud pattern detection across 15 pattern types including
          duplicate reuse, quantity tampering, date manipulation,
          serial anomaly, template forgery, geo-impossibility,
          velocity anomaly, round-number bias, and copy-paste
          detection
        - Cross-reference verification against FSC, RSPO, ISCC,
          Fairtrade, UTZ/RA, IPPC, and national customs registries
          with rate limiting and caching
        - Compliance reporting in JSON, PDF, CSV, and EUDR XML
          formats with evidence package generation

    Foundational modules:
        - config: DocumentAuthenticationConfig with GL_EUDR_DAV_
          env var support (40+ settings)
        - models: Pydantic v2 data models with 15 enumerations,
          9 core models, 15 request models, and 15 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_dav_)

PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Standard: eIDAS Regulation (EU) No 910/2014 for digital signatures
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.document_authentication import (
    ...     DocumentRecord,
    ...     DocumentType,
    ...     ClassificationConfidence,
    ...     SignatureStandard,
    ...     DocumentAuthenticationConfig,
    ...     get_config,
    ... )
    >>> doc = DocumentRecord(
    ...     document_id="doc-001",
    ...     file_name="certificate_of_origin.pdf",
    ...     file_hash_sha256="abc123...",
    ...     document_type=DocumentType.COO,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-DAV-012"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.document_authentication.config import (
        DocumentAuthenticationConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    DocumentAuthenticationConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.document_authentication.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_HASH_ALGORITHMS,
        SUPPORTED_SIGNATURE_STANDARDS,
        MIN_RSA_KEY_SIZE,
        MIN_ECDSA_KEY_SIZE,
        # Enumerations
        DocumentType,
        ClassificationConfidence,
        SignatureStandard,
        SignatureStatus,
        HashAlgorithm,
        CertificateStatus,
        FraudSeverity,
        FraudPatternType,
        VerificationStatus,
        RegistryType,
        ReportFormat,
        MetadataField,
        DocumentLanguage,
        AuthenticationResult,
        BatchJobStatus,
        # Core Models
        DocumentRecord,
        ClassificationResult,
        SignatureVerificationResult,
        HashRecord,
        CertificateChainResult,
        MetadataRecord,
        FraudAlert,
        CrossRefResult,
        AuthenticationReport,
        # Request Models
        ClassifyDocumentRequest,
        BatchClassifyRequest,
        VerifySignatureRequest,
        ComputeHashRequest,
        VerifyHashRequest,
        ValidateCertificateRequest,
        ExtractMetadataRequest,
        DetectFraudRequest,
        CrossRefVerifyRequest,
        GenerateReportRequest,
        BatchVerificationRequest,
        RegisterTemplateRequest,
        AddTrustedCARequest,
        SearchDocumentsRequest,
        GetFraudAlertsRequest,
        # Response Models
        ClassificationResponse,
        SignatureResponse,
        HashResponse,
        CertificateResponse,
        MetadataResponse,
        FraudDetectionResponse,
        CrossRefResponse,
        ReportResponse,
        BatchResponse,
        HealthResponse,
        DashboardResponse,
        TemplateResponse,
        TrustedCAResponse,
        DocumentSearchResponse,
        FraudAlertListResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.document_authentication.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.document_authentication.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        dav_documents_processed_total,
        dav_classifications_total,
        dav_signatures_verified_total,
        dav_signatures_invalid_total,
        dav_hashes_computed_total,
        dav_duplicates_detected_total,
        dav_tampering_detected_total,
        dav_cert_chains_validated_total,
        dav_cert_revocations_total,
        dav_fraud_alerts_total,
        dav_fraud_critical_total,
        dav_crossref_queries_total,
        dav_reports_generated_total,
        dav_api_errors_total,
        dav_classification_duration_seconds,
        dav_verification_duration_seconds,
        dav_crossref_duration_seconds,
        dav_active_verifications,
        # Helper functions
        record_document_processed,
        record_classification,
        record_signature_verified,
        record_signature_invalid,
        record_hash_computed,
        record_duplicate_detected,
        record_tampering_detected,
        record_cert_chain_validated,
        record_cert_revocation,
        record_fraud_alert,
        record_fraud_critical,
        record_crossref_query,
        record_report_generated,
        record_api_error,
        observe_classification_duration,
        observe_verification_duration,
        observe_crossref_duration,
        set_active_verifications,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "DocumentAuthenticationConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_HASH_ALGORITHMS",
    "SUPPORTED_SIGNATURE_STANDARDS",
    "MIN_RSA_KEY_SIZE",
    "MIN_ECDSA_KEY_SIZE",
    # -- Enumerations --
    "DocumentType",
    "ClassificationConfidence",
    "SignatureStandard",
    "SignatureStatus",
    "HashAlgorithm",
    "CertificateStatus",
    "FraudSeverity",
    "FraudPatternType",
    "VerificationStatus",
    "RegistryType",
    "ReportFormat",
    "MetadataField",
    "DocumentLanguage",
    "AuthenticationResult",
    "BatchJobStatus",
    # -- Core Models --
    "DocumentRecord",
    "ClassificationResult",
    "SignatureVerificationResult",
    "HashRecord",
    "CertificateChainResult",
    "MetadataRecord",
    "FraudAlert",
    "CrossRefResult",
    "AuthenticationReport",
    # -- Request Models --
    "ClassifyDocumentRequest",
    "BatchClassifyRequest",
    "VerifySignatureRequest",
    "ComputeHashRequest",
    "VerifyHashRequest",
    "ValidateCertificateRequest",
    "ExtractMetadataRequest",
    "DetectFraudRequest",
    "CrossRefVerifyRequest",
    "GenerateReportRequest",
    "BatchVerificationRequest",
    "RegisterTemplateRequest",
    "AddTrustedCARequest",
    "SearchDocumentsRequest",
    "GetFraudAlertsRequest",
    # -- Response Models --
    "ClassificationResponse",
    "SignatureResponse",
    "HashResponse",
    "CertificateResponse",
    "MetadataResponse",
    "FraudDetectionResponse",
    "CrossRefResponse",
    "ReportResponse",
    "BatchResponse",
    "HealthResponse",
    "DashboardResponse",
    "TemplateResponse",
    "TrustedCAResponse",
    "DocumentSearchResponse",
    "FraudAlertListResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "dav_documents_processed_total",
    "dav_classifications_total",
    "dav_signatures_verified_total",
    "dav_signatures_invalid_total",
    "dav_hashes_computed_total",
    "dav_duplicates_detected_total",
    "dav_tampering_detected_total",
    "dav_cert_chains_validated_total",
    "dav_cert_revocations_total",
    "dav_fraud_alerts_total",
    "dav_fraud_critical_total",
    "dav_crossref_queries_total",
    "dav_reports_generated_total",
    "dav_api_errors_total",
    "dav_classification_duration_seconds",
    "dav_verification_duration_seconds",
    "dav_crossref_duration_seconds",
    "dav_active_verifications",
    "record_document_processed",
    "record_classification",
    "record_signature_verified",
    "record_signature_invalid",
    "record_hash_computed",
    "record_duplicate_detected",
    "record_tampering_detected",
    "record_cert_chain_validated",
    "record_cert_revocation",
    "record_fraud_alert",
    "record_fraud_critical",
    "record_crossref_query",
    "record_report_generated",
    "record_api_error",
    "observe_classification_duration",
    "observe_verification_duration",
    "observe_crossref_duration",
    "set_active_verifications",
]
