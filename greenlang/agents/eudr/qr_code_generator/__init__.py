# -*- coding: utf-8 -*-
"""
QR Code Generator Agent - AGENT-EUDR-014

Production-grade QR code generation engine for EUDR compliance covering
QR code creation with configurable version (auto/v1-v40), error correction
(L/M/Q/H), output format (PNG/SVG/PDF/ZPL/EPS), and ISO/IEC 18004
quality grading; data payload composition with zlib compression and
AES-256-GCM encryption; label rendering with EUDR compliance status
colour coding and five template types (product/shipping/pallet/
container/consumer); batch code generation with Luhn/ISO 7064/CRC-8
check digits; verification URL construction with HMAC-SHA256 signed
tokens and configurable TTL; anti-counterfeiting via scan velocity
monitoring, geo-fencing, and digital watermarking; bulk generation
job orchestration for up to 100,000 codes per job; QR code lifecycle
management (create/activate/deactivate/revoke/expire); scan event
recording with counterfeit risk assessment; and comprehensive
audit logging.

This package provides a complete QR code generation system for
EUDR supply chain traceability supporting tamper-evident compliance
labels per EU 2023/1115 Articles 4, 10, and 14:

    Capabilities:
        - QR code generation with automatic or fixed version selection
          per ISO/IEC 18004, four error correction levels, five output
          formats, configurable module size and quiet zone, optional
          centre logo embedding, and ISO/IEC 15416 quality grading
        - Data payload composition with five content types
          (full_traceability, compact_verification, consumer_summary,
          batch_identifier, blockchain_anchor), zlib compression for
          payloads exceeding configurable threshold, and optional
          AES-256-GCM encryption for sensitive data
        - Label rendering with five pre-designed templates, EUDR
          compliance status colour coding (green/amber/red), custom
          fields, configurable font and DPI, and print bleed margins
        - Batch code generation with operator-commodity-year prefix
          format, three check digit algorithms (Luhn, ISO 7064 Mod
          11,10, CRC-8), configurable zero-padding and sequence start
        - Verification URL construction with HMAC-SHA256 signed tokens,
          configurable truncation length, 5-year TTL per EUDR Article
          14, and optional short URL integration
        - Anti-counterfeiting with HMAC secret key rotation, digital
          watermark embedding, scan velocity threshold monitoring
          (100 scans/min default), and geographic fence enforcement
        - Bulk QR code generation for high-volume operations with
          configurable worker count, timeout, ZIP packaging, and
          post-generation validation
        - QR code lifecycle management with five statuses (created,
          active, deactivated, revoked, expired), reprint tracking,
          and scan count monitoring
        - Scan event recording with counterfeit risk assessment
          (low/medium/high/critical), geo-location capture, HMAC
          token validation, and velocity anomaly detection
        - Four symbology types: QR Code, Micro QR, Data Matrix,
          and GS1 Digital Link for varied supply chain applications

    Foundational modules:
        - config: QRCodeGeneratorConfig with GL_EUDR_QRG_
          env var support (50+ settings)
        - models: Pydantic v2 data models with 15 enumerations,
          12 core models, 15 request models, and 15 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_qrg_)

PRD: PRD-AGENT-EUDR-014
Agent ID: GL-EUDR-QRG-014
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.qr_code_generator import (
    ...     QRCodeRecord,
    ...     ContentType,
    ...     ErrorCorrectionLevel,
    ...     OutputFormat,
    ...     QRCodeGeneratorConfig,
    ...     get_config,
    ... )
    >>> code = QRCodeRecord(
    ...     payload_hash="a" * 64,
    ...     payload_size_bytes=256,
    ...     content_type=ContentType.COMPACT_VERIFICATION,
    ...     operator_id="operator-001",
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-QRG-014"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.qr_code_generator.config import (
        QRCodeGeneratorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    QRCodeGeneratorConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.qr_code_generator.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        MAX_QR_PAYLOAD_BYTES,
        MAX_QR_VERSION,
        SUPPORTED_OUTPUT_FORMATS,
        SUPPORTED_CONTENT_TYPES,
        SUPPORTED_SYMBOLOGY_TYPES,
        SUPPORTED_LABEL_TEMPLATES,
        SUPPORTED_CHECK_DIGIT_ALGORITHMS,
        DEFAULT_EUDR_COMMODITIES,
        # Enumerations
        QRCodeVersion,
        ErrorCorrectionLevel,
        OutputFormat,
        ContentType,
        SymbologyType,
        LabelTemplate,
        CheckDigitAlgorithm,
        CodeStatus,
        ScanOutcome,
        CounterfeitRiskLevel,
        BulkJobStatus,
        EUDRCommodity,
        ComplianceStatus,
        PayloadEncoding,
        DPILevel,
        # Core Models
        QRCodeRecord,
        DataPayload,
        LabelRecord,
        BatchCode,
        VerificationURL,
        SignatureRecord,
        ScanEvent,
        BulkJob,
        LifecycleEvent,
        TemplateDefinition,
        CodeAssociation,
        AuditLogEntry,
        # Request Models
        GenerateQRCodeRequest,
        ComposePayloadRequest,
        RenderLabelRequest,
        GenerateBatchCodeRequest,
        BuildVerificationURLRequest,
        SignCodeRequest,
        RecordScanRequest,
        SubmitBulkJobRequest,
        ActivateCodeRequest,
        DeactivateCodeRequest,
        RevokeCodeRequest,
        ReprintCodeRequest,
        SearchCodesRequest,
        GetScanHistoryRequest,
        ValidateCodeRequest,
        # Response Models
        QRCodeResponse,
        PayloadResponse,
        LabelResponse,
        BatchCodeResponse,
        VerificationURLResponse,
        SignatureResponse,
        ScanResponse,
        BulkJobResponse,
        ActivateResponse,
        DeactivateResponse,
        RevokeResponse,
        ReprintResponse,
        SearchResponse,
        ScanHistoryResponse,
        HealthResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.qr_code_generator.provenance import (
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
    from greenlang.agents.eudr.qr_code_generator.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        qrg_codes_generated_total,
        qrg_labels_generated_total,
        qrg_payloads_composed_total,
        qrg_batch_codes_total,
        qrg_verification_urls_total,
        qrg_scans_total,
        qrg_counterfeit_detections_total,
        qrg_bulk_jobs_total,
        qrg_bulk_codes_total,
        qrg_revocations_total,
        qrg_signature_verifications_total,
        qrg_api_errors_total,
        qrg_generation_duration_seconds,
        qrg_label_duration_seconds,
        qrg_bulk_duration_seconds,
        qrg_verification_duration_seconds,
        qrg_active_bulk_jobs,
        qrg_active_codes,
        # Helper functions
        record_code_generated,
        record_label_generated,
        record_payload_composed,
        record_batch_code_generated,
        record_verification_url_built,
        record_scan,
        record_counterfeit_detection,
        record_bulk_job,
        record_bulk_codes,
        record_revocation,
        record_signature_verification,
        record_api_error,
        observe_generation_duration,
        observe_label_duration,
        observe_bulk_duration,
        observe_verification_duration,
        set_active_bulk_jobs,
        set_active_codes,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 1: QR Encoder ----
try:
    from greenlang.agents.eudr.qr_code_generator.qr_encoder import QREncoder
except ImportError:
    QREncoder = None  # type: ignore[assignment,misc]

# ---- Engine 2: Payload Composer ----
try:
    from greenlang.agents.eudr.qr_code_generator.payload_composer import PayloadComposer
except ImportError:
    PayloadComposer = None  # type: ignore[assignment,misc]

# ---- Engine 3: Label Template Engine ----
try:
    from greenlang.agents.eudr.qr_code_generator.label_template_engine import LabelTemplateEngine
except ImportError:
    LabelTemplateEngine = None  # type: ignore[assignment,misc]

# ---- Engine 4: Batch Code Generator ----
try:
    from greenlang.agents.eudr.qr_code_generator.batch_code_generator import BatchCodeGenerator
except ImportError:
    BatchCodeGenerator = None  # type: ignore[assignment,misc]

# ---- Engine 5: Verification URL Builder ----
try:
    from greenlang.agents.eudr.qr_code_generator.verification_url_builder import VerificationURLBuilder
except ImportError:
    VerificationURLBuilder = None  # type: ignore[assignment,misc]

# ---- Engine 6: Anti-Counterfeit Engine ----
try:
    from greenlang.agents.eudr.qr_code_generator.anti_counterfeit_engine import AntiCounterfeitEngine
except ImportError:
    AntiCounterfeitEngine = None  # type: ignore[assignment,misc]

# ---- Engine 7: Bulk Generation Pipeline ----
try:
    from greenlang.agents.eudr.qr_code_generator.bulk_generation_pipeline import BulkGenerationPipeline
except ImportError:
    BulkGenerationPipeline = None  # type: ignore[assignment,misc]

# ---- Engine 8: Code Lifecycle Manager ----
try:
    from greenlang.agents.eudr.qr_code_generator.code_lifecycle_manager import CodeLifecycleManager
except ImportError:
    CodeLifecycleManager = None  # type: ignore[assignment,misc]

# ---- Service Facade ----
try:
    from greenlang.agents.eudr.qr_code_generator.setup import QRCodeGeneratorService
except ImportError:
    QRCodeGeneratorService = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "QRCodeGeneratorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "MAX_QR_PAYLOAD_BYTES",
    "MAX_QR_VERSION",
    "SUPPORTED_OUTPUT_FORMATS",
    "SUPPORTED_CONTENT_TYPES",
    "SUPPORTED_SYMBOLOGY_TYPES",
    "SUPPORTED_LABEL_TEMPLATES",
    "SUPPORTED_CHECK_DIGIT_ALGORITHMS",
    "DEFAULT_EUDR_COMMODITIES",
    # -- Enumerations --
    "QRCodeVersion",
    "ErrorCorrectionLevel",
    "OutputFormat",
    "ContentType",
    "SymbologyType",
    "LabelTemplate",
    "CheckDigitAlgorithm",
    "CodeStatus",
    "ScanOutcome",
    "CounterfeitRiskLevel",
    "BulkJobStatus",
    "EUDRCommodity",
    "ComplianceStatus",
    "PayloadEncoding",
    "DPILevel",
    # -- Core Models --
    "QRCodeRecord",
    "DataPayload",
    "LabelRecord",
    "BatchCode",
    "VerificationURL",
    "SignatureRecord",
    "ScanEvent",
    "BulkJob",
    "LifecycleEvent",
    "TemplateDefinition",
    "CodeAssociation",
    "AuditLogEntry",
    # -- Request Models --
    "GenerateQRCodeRequest",
    "ComposePayloadRequest",
    "RenderLabelRequest",
    "GenerateBatchCodeRequest",
    "BuildVerificationURLRequest",
    "SignCodeRequest",
    "RecordScanRequest",
    "SubmitBulkJobRequest",
    "ActivateCodeRequest",
    "DeactivateCodeRequest",
    "RevokeCodeRequest",
    "ReprintCodeRequest",
    "SearchCodesRequest",
    "GetScanHistoryRequest",
    "ValidateCodeRequest",
    # -- Response Models --
    "QRCodeResponse",
    "PayloadResponse",
    "LabelResponse",
    "BatchCodeResponse",
    "VerificationURLResponse",
    "SignatureResponse",
    "ScanResponse",
    "BulkJobResponse",
    "ActivateResponse",
    "DeactivateResponse",
    "RevokeResponse",
    "ReprintResponse",
    "SearchResponse",
    "ScanHistoryResponse",
    "HealthResponse",
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
    "qrg_codes_generated_total",
    "qrg_labels_generated_total",
    "qrg_payloads_composed_total",
    "qrg_batch_codes_total",
    "qrg_verification_urls_total",
    "qrg_scans_total",
    "qrg_counterfeit_detections_total",
    "qrg_bulk_jobs_total",
    "qrg_bulk_codes_total",
    "qrg_revocations_total",
    "qrg_signature_verifications_total",
    "qrg_api_errors_total",
    "qrg_generation_duration_seconds",
    "qrg_label_duration_seconds",
    "qrg_bulk_duration_seconds",
    "qrg_verification_duration_seconds",
    "qrg_active_bulk_jobs",
    "qrg_active_codes",
    "record_code_generated",
    "record_label_generated",
    "record_payload_composed",
    "record_batch_code_generated",
    "record_verification_url_built",
    "record_scan",
    "record_counterfeit_detection",
    "record_bulk_job",
    "record_bulk_codes",
    "record_revocation",
    "record_signature_verification",
    "record_api_error",
    "observe_generation_duration",
    "observe_label_duration",
    "observe_bulk_duration",
    "observe_verification_duration",
    "set_active_bulk_jobs",
    "set_active_codes",
    # -- Engines --
    "QREncoder",
    "PayloadComposer",
    "LabelTemplateEngine",
    "BatchCodeGenerator",
    "VerificationURLBuilder",
    "AntiCounterfeitEngine",
    "BulkGenerationPipeline",
    "CodeLifecycleManager",
    # -- Service Facade --
    "QRCodeGeneratorService",
]
