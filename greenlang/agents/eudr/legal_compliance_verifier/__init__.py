# -*- coding: utf-8 -*-
"""
Legal Compliance Verifier Agent - AGENT-EUDR-023

Production-grade legal compliance verification platform for EUDR compliance
covering legal framework database management across EUDR Article 2(40)
eight legislation categories (land_use_rights, environmental_protection,
forest_related_rules, third_party_rights, labour_rights, tax_and_royalty,
trade_and_customs, anti_corruption); document verification with 4-step
pipeline (format_validation, issuer_verification, validity_check,
scope_alignment) across 16 document types; certification scheme validation
for 5 schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC) with EUDR
equivalence mapping and 10-point validation checklist; red flag detection
evaluating 40 indicators across 6 categories (documentation, supply_chain,
geographic, certification, financial, operational) with deterministic
weighted scoring using country and commodity multipliers; country compliance
checking across 27 commodity-producing countries with per-category
assessment and combined risk scoring (60% compliance + 40% red flag risk);
third-party audit report processing for 7 audit types (FSC, PEFC, RSPO,
EUDR_DD, government_inspection, ISO_14001, ISO_45001) with finding
normalization and corrective action tracking; and compliance reporting in
8 report types, 5 output formats (PDF/JSON/HTML/XBRL/XML), and 5
languages (EN/FR/DE/ES/PT).

This package provides a complete legal compliance verification system for
EUDR regulatory compliance per EU 2023/1115 Articles 2(40), 10, 11, 12,
13, 14, 29, and 31:

    Capabilities:
        - Legal framework database with 27 country coverage, 8 EUDR
          Article 2(40) legislation categories, framework applicability
          scoring, coverage matrix generation, and gap analysis with
          remediation recommendations
        - Document verification with 4-step pipeline (format validation,
          issuer verification, validity check, scope alignment), 16
          document types mapped to legislation categories, weighted
          scoring (documents_present=0.40, document_validity=0.30,
          scope_alignment=0.20, authenticity=0.10), and expiry
          monitoring at 30/60/90-day warning thresholds
        - Certification scheme validation for FSC, PEFC, RSPO,
          Rainforest Alliance, and ISCC with 10-point validation
          checklist (cert_format, cb_accreditation, scope_validation,
          validity_period, coc_model, surveillance_audit,
          non_conformities, suspension_status, eudr_equivalence,
          multi_site_scope) and EUDR equivalence scoring
        - Red flag detection evaluating 40 indicators across 6 risk
          categories with deterministic scoring formula:
          flag_score = base_weight * country_multiplier * commodity_multiplier;
          aggregate = (SUM(flag_scores) / max_possible) * 100
        - Country compliance checking with per-category assessment,
          combined risk scoring (60% compliance risk + 40% red flag risk),
          and compliance status classification (COMPLIANT>=80,
          PARTIALLY_COMPLIANT>=50, NON_COMPLIANT<50)
        - Third-party audit processing for 7 audit types with finding
          normalization (5 severity categories), compliance impact
          scoring (60% conclusion + 40% findings severity), and
          corrective action deadline tracking (major_NC=60d, minor_NC=90d)
        - Compliance reporting in 8 report types with format-specific
          constraints, S3 storage key computation, file size estimation,
          and multi-language label translation

    Foundational modules:
        - config: LegalComplianceVerifierConfig with GL_EUDR_LCV_
          env var support (90+ settings)
        - models: Pydantic v2 data models with 12 enumerations,
          10 core models, 12 request models, and 12 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 20 Prometheus self-monitoring metrics (gl_eudr_lcv_)

PRD: PRD-AGENT-EUDR-023
Agent ID: GL-EUDR-LCV-023
Regulation: EU 2023/1115 (EUDR) Articles 2(40), 10, 11, 12, 13, 14, 29, 31
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.legal_compliance_verifier import (
    ...     LegislationCategory,
    ...     ComplianceStatus,
    ...     RiskLevel,
    ...     CommodityType,
    ...     LegalComplianceVerifierConfig,
    ...     get_config,
    ... )
    >>> from decimal import Decimal
    >>> cfg = get_config()
    >>> print(cfg.compliant_threshold, cfg.partial_threshold)
    80 50
    >>> cat = LegislationCategory.FOREST_RELATED_RULES
    >>> print(cat.value)
    forest_related_rules

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-LCV-023"

# ---------------------------------------------------------------------------
# Foundational imports: config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.config import (
        LegalComplianceVerifierConfig,
        get_config,
        reset_config,
        set_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import config module: {e}")
    LegalComplianceVerifierConfig = None  # type: ignore[misc,assignment]
    get_config = None  # type: ignore[misc,assignment]
    reset_config = None  # type: ignore[misc,assignment]
    set_config = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: models
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.models import (
        # Enumerations (12)
        LegislationCategory,
        ComplianceStatus,
        DocumentValidityStatus,
        CertificationScheme,
        RedFlagSeverity,
        RedFlagCategory,
        AuditFindingCategory,
        AuditFindingStatus,
        ReportType,
        ReportFormat,
        RiskLevel,
        CommodityType,
        # Core Models (10)
        LegalFramework,
        ComplianceDocument,
        CertificationRecord,
        RedFlagAlert,
        AuditFinding,
        AuditReport,
        ComplianceAssessment,
        LegalRequirement,
        ComplianceReport,
        AuditLogEntry,
        # Request Models (12)
        QueryLegalFrameworkRequest,
        VerifyDocumentRequest,
        ValidateCertificationRequest,
        ScanRedFlagsRequest,
        AssessComplianceRequest,
        SubmitAuditReportRequest,
        GenerateReportRequest,
        BatchAssessmentRequest,
        AcknowledgeRedFlagRequest,
        UpdateFrameworkRequest,
        ExpiringDocumentsRequest,
        CountryComplianceRequest,
        # Response Models (12)
        LegalFrameworkResponse,
        DocumentVerificationResponse,
        CertificationValidationResponse,
        RedFlagScanResponse,
        ComplianceAssessmentResponse,
        AuditReportResponse,
        ComplianceReportResponse,
        BatchAssessmentResponse,
        ExpiringDocumentsResponse,
        CountryComplianceSummaryResponse,
        HealthCheckResponse,
        AdminStatsResponse,
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import models module: {e}")
    # Enumerations (12)
    LegislationCategory = None  # type: ignore[misc,assignment]
    ComplianceStatus = None  # type: ignore[misc,assignment]
    DocumentValidityStatus = None  # type: ignore[misc,assignment]
    CertificationScheme = None  # type: ignore[misc,assignment]
    RedFlagSeverity = None  # type: ignore[misc,assignment]
    RedFlagCategory = None  # type: ignore[misc,assignment]
    AuditFindingCategory = None  # type: ignore[misc,assignment]
    AuditFindingStatus = None  # type: ignore[misc,assignment]
    ReportType = None  # type: ignore[misc,assignment]
    ReportFormat = None  # type: ignore[misc,assignment]
    RiskLevel = None  # type: ignore[misc,assignment]
    CommodityType = None  # type: ignore[misc,assignment]
    # Core Models (10)
    LegalFramework = None  # type: ignore[misc,assignment]
    ComplianceDocument = None  # type: ignore[misc,assignment]
    CertificationRecord = None  # type: ignore[misc,assignment]
    RedFlagAlert = None  # type: ignore[misc,assignment]
    AuditFinding = None  # type: ignore[misc,assignment]
    AuditReport = None  # type: ignore[misc,assignment]
    ComplianceAssessment = None  # type: ignore[misc,assignment]
    LegalRequirement = None  # type: ignore[misc,assignment]
    ComplianceReport = None  # type: ignore[misc,assignment]
    AuditLogEntry = None  # type: ignore[misc,assignment]
    # Request Models (12)
    QueryLegalFrameworkRequest = None  # type: ignore[misc,assignment]
    VerifyDocumentRequest = None  # type: ignore[misc,assignment]
    ValidateCertificationRequest = None  # type: ignore[misc,assignment]
    ScanRedFlagsRequest = None  # type: ignore[misc,assignment]
    AssessComplianceRequest = None  # type: ignore[misc,assignment]
    SubmitAuditReportRequest = None  # type: ignore[misc,assignment]
    GenerateReportRequest = None  # type: ignore[misc,assignment]
    BatchAssessmentRequest = None  # type: ignore[misc,assignment]
    AcknowledgeRedFlagRequest = None  # type: ignore[misc,assignment]
    UpdateFrameworkRequest = None  # type: ignore[misc,assignment]
    ExpiringDocumentsRequest = None  # type: ignore[misc,assignment]
    CountryComplianceRequest = None  # type: ignore[misc,assignment]
    # Response Models (12)
    LegalFrameworkResponse = None  # type: ignore[misc,assignment]
    DocumentVerificationResponse = None  # type: ignore[misc,assignment]
    CertificationValidationResponse = None  # type: ignore[misc,assignment]
    RedFlagScanResponse = None  # type: ignore[misc,assignment]
    ComplianceAssessmentResponse = None  # type: ignore[misc,assignment]
    AuditReportResponse = None  # type: ignore[misc,assignment]
    ComplianceReportResponse = None  # type: ignore[misc,assignment]
    BatchAssessmentResponse = None  # type: ignore[misc,assignment]
    ExpiringDocumentsResponse = None  # type: ignore[misc,assignment]
    CountryComplianceSummaryResponse = None  # type: ignore[misc,assignment]
    HealthCheckResponse = None  # type: ignore[misc,assignment]
    AdminStatsResponse = None  # type: ignore[misc,assignment]
    # Constants
    VERSION = None  # type: ignore[misc,assignment]
    EUDR_CUTOFF_DATE = None  # type: ignore[misc,assignment]
    MAX_BATCH_SIZE = None  # type: ignore[misc,assignment]
    EUDR_RETENTION_YEARS = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_tracker,
        reset_tracker,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import provenance module: {e}")
    ProvenanceRecord = None  # type: ignore[misc,assignment]
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_tracker = None  # type: ignore[misc,assignment]
    reset_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.metrics import (
        PROMETHEUS_AVAILABLE,
        # Counter helpers (10)
        record_framework_query,
        record_document_verification,
        record_certification_validation,
        record_red_flag_scan,
        record_red_flag_triggered,
        record_compliance_assessment,
        record_audit_report_processed,
        record_report_generated,
        record_batch_job,
        record_api_error,
        # Histogram helpers (5)
        observe_compliance_check_duration,
        observe_full_assessment_duration,
        observe_document_verification_duration,
        observe_red_flag_scan_duration,
        observe_report_generation_duration,
        # Gauge helpers (5)
        set_countries_covered,
        set_active_red_flags,
        set_expiring_documents_30d,
        set_non_compliant_suppliers,
        set_cache_hit_ratio,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import metrics module: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    # Counter helpers (10)
    record_framework_query = None  # type: ignore[misc,assignment]
    record_document_verification = None  # type: ignore[misc,assignment]
    record_certification_validation = None  # type: ignore[misc,assignment]
    record_red_flag_scan = None  # type: ignore[misc,assignment]
    record_red_flag_triggered = None  # type: ignore[misc,assignment]
    record_compliance_assessment = None  # type: ignore[misc,assignment]
    record_audit_report_processed = None  # type: ignore[misc,assignment]
    record_report_generated = None  # type: ignore[misc,assignment]
    record_batch_job = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]
    # Histogram helpers (5)
    observe_compliance_check_duration = None  # type: ignore[misc,assignment]
    observe_full_assessment_duration = None  # type: ignore[misc,assignment]
    observe_document_verification_duration = None  # type: ignore[misc,assignment]
    observe_red_flag_scan_duration = None  # type: ignore[misc,assignment]
    observe_report_generation_duration = None  # type: ignore[misc,assignment]
    # Gauge helpers (5)
    set_countries_covered = None  # type: ignore[misc,assignment]
    set_active_red_flags = None  # type: ignore[misc,assignment]
    set_expiring_documents_30d = None  # type: ignore[misc,assignment]
    set_non_compliant_suppliers = None  # type: ignore[misc,assignment]
    set_cache_hit_ratio = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: Legal Framework Database Engine ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.legal_framework_database_engine import (
        LegalFrameworkDatabaseEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.legal_framework_database_engine import (
            LegalFrameworkDatabaseEngine,
        )
    except ImportError:
        LegalFrameworkDatabaseEngine = None  # type: ignore[misc,assignment]

# ---- Engine 2: Document Verification Engine ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.document_verification_engine import (
        DocumentVerificationEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.document_verification_engine import (
            DocumentVerificationEngine,
        )
    except ImportError:
        DocumentVerificationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 3: Certification Scheme Validator ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.certification_scheme_validator import (
        CertificationSchemeValidator,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.certification_scheme_validator import (
            CertificationSchemeValidator,
        )
    except ImportError:
        CertificationSchemeValidator = None  # type: ignore[misc,assignment]

# ---- Engine 4: Red Flag Detection Engine ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.red_flag_detection_engine import (
        RedFlagDetectionEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.red_flag_detection_engine import (
            RedFlagDetectionEngine,
        )
    except ImportError:
        RedFlagDetectionEngine = None  # type: ignore[misc,assignment]

# ---- Engine 5: Country Compliance Checker ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.country_compliance_checker import (
        CountryComplianceChecker,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.country_compliance_checker import (
            CountryComplianceChecker,
        )
    except ImportError:
        CountryComplianceChecker = None  # type: ignore[misc,assignment]

# ---- Engine 6: Third-Party Audit Engine ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.third_party_audit_engine import (
        ThirdPartyAuditEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.third_party_audit_engine import (
            ThirdPartyAuditEngine,
        )
    except ImportError:
        ThirdPartyAuditEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Compliance Reporting Engine ----
try:
    from greenlang.agents.eudr.legal_compliance_verifier.compliance_reporting_engine import (
        ComplianceReportingEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.legal_compliance_verifier.engines.compliance_reporting_engine import (
            ComplianceReportingEngine,
        )
    except ImportError:
        ComplianceReportingEngine = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Service facade import (conditional - service may not exist yet)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.setup import (
        LegalComplianceVerifierSetup,
        get_service,
        set_service,
        reset_service,
        lifespan,
    )
except ImportError:
    LegalComplianceVerifierSetup = None  # type: ignore[misc,assignment]
    get_service = None  # type: ignore[misc,assignment]
    set_service = None  # type: ignore[misc,assignment]
    reset_service = None  # type: ignore[misc,assignment]
    lifespan = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Reference data imports (conditional)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.reference_data import (
        ARTICLE_2_40_CATEGORIES,
        COUNTRY_FRAMEWORKS,
        CERTIFICATION_SCHEMES,
        EUDR_EQUIVALENCE_MATRIX,
        RED_FLAG_INDICATORS,
    )
except ImportError:
    ARTICLE_2_40_CATEGORIES = None  # type: ignore[misc,assignment]
    COUNTRY_FRAMEWORKS = None  # type: ignore[misc,assignment]
    CERTIFICATION_SCHEMES = None  # type: ignore[misc,assignment]
    EUDR_EQUIVALENCE_MATRIX = None  # type: ignore[misc,assignment]
    RED_FLAG_INDICATORS = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Config --
    "LegalComplianceVerifierConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Enumerations (12) --
    "LegislationCategory",
    "ComplianceStatus",
    "DocumentValidityStatus",
    "CertificationScheme",
    "RedFlagSeverity",
    "RedFlagCategory",
    "AuditFindingCategory",
    "AuditFindingStatus",
    "ReportType",
    "ReportFormat",
    "RiskLevel",
    "CommodityType",
    # -- Core Models (10) --
    "LegalFramework",
    "ComplianceDocument",
    "CertificationRecord",
    "RedFlagAlert",
    "AuditFinding",
    "AuditReport",
    "ComplianceAssessment",
    "LegalRequirement",
    "ComplianceReport",
    "AuditLogEntry",
    # -- Request Models (12) --
    "QueryLegalFrameworkRequest",
    "VerifyDocumentRequest",
    "ValidateCertificationRequest",
    "ScanRedFlagsRequest",
    "AssessComplianceRequest",
    "SubmitAuditReportRequest",
    "GenerateReportRequest",
    "BatchAssessmentRequest",
    "AcknowledgeRedFlagRequest",
    "UpdateFrameworkRequest",
    "ExpiringDocumentsRequest",
    "CountryComplianceRequest",
    # -- Response Models (12) --
    "LegalFrameworkResponse",
    "DocumentVerificationResponse",
    "CertificationValidationResponse",
    "RedFlagScanResponse",
    "ComplianceAssessmentResponse",
    "AuditReportResponse",
    "ComplianceReportResponse",
    "BatchAssessmentResponse",
    "ExpiringDocumentsResponse",
    "CountryComplianceSummaryResponse",
    "HealthCheckResponse",
    "AdminStatsResponse",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_tracker",
    "reset_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_framework_query",
    "record_document_verification",
    "record_certification_validation",
    "record_red_flag_scan",
    "record_red_flag_triggered",
    "record_compliance_assessment",
    "record_audit_report_processed",
    "record_report_generated",
    "record_batch_job",
    "record_api_error",
    "observe_compliance_check_duration",
    "observe_full_assessment_duration",
    "observe_document_verification_duration",
    "observe_red_flag_scan_duration",
    "observe_report_generation_duration",
    "set_countries_covered",
    "set_active_red_flags",
    "set_expiring_documents_30d",
    "set_non_compliant_suppliers",
    "set_cache_hit_ratio",
    # -- Engines (7) --
    "LegalFrameworkDatabaseEngine",
    "DocumentVerificationEngine",
    "CertificationSchemeValidator",
    "RedFlagDetectionEngine",
    "CountryComplianceChecker",
    "ThirdPartyAuditEngine",
    "ComplianceReportingEngine",
    # -- Service Facade --
    "LegalComplianceVerifierSetup",
    "get_service",
    "set_service",
    "reset_service",
    "lifespan",
    # -- Reference Data --
    "ARTICLE_2_40_CATEGORIES",
    "COUNTRY_FRAMEWORKS",
    "CERTIFICATION_SCHEMES",
    "EUDR_EQUIVALENCE_MATRIX",
    "RED_FLAG_INDICATORS",
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, and model counts for the Legal Compliance
        Verifier agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-LCV-023'
        >>> info["engine_count"]
        7
        >>> info["legislation_categories"]
        ['land_use_rights', 'environmental_protection', ...]
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Legal Compliance Verifier",
        "prd": "PRD-AGENT-EUDR-023",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["2(40)", "10", "11", "12", "13", "14", "29", "31"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "legislation_categories": [
            "land_use_rights",
            "environmental_protection",
            "forest_related_rules",
            "third_party_rights",
            "labour_rights",
            "tax_and_royalty",
            "trade_and_customs",
            "anti_corruption",
        ],
        "commodities": [
            "cattle",
            "cocoa",
            "coffee",
            "oil_palm",
            "rubber",
            "soya",
            "wood",
        ],
        "certification_schemes": [
            "FSC",
            "PEFC",
            "RSPO",
            "Rainforest Alliance",
            "ISCC",
        ],
        "countries_covered": 27,
        "red_flag_indicators": 40,
        "red_flag_categories": [
            "documentation",
            "supply_chain",
            "geographic",
            "certification",
            "financial",
            "operational",
        ],
        "audit_types": [
            "FSC",
            "PEFC",
            "RSPO",
            "EUDR_DD",
            "government_inspection",
            "ISO_14001",
            "ISO_45001",
        ],
        "report_types": [
            "full_assessment",
            "category_specific",
            "supplier_scorecard",
            "red_flag_summary",
            "document_status",
            "certification_validity",
            "country_framework",
            "dds_annex",
        ],
        "report_formats": ["pdf", "json", "html", "xbrl", "xml"],
        "report_languages": ["en", "fr", "de", "es", "pt"],
        "engines": [
            "LegalFrameworkDatabaseEngine",
            "DocumentVerificationEngine",
            "CertificationSchemeValidator",
            "RedFlagDetectionEngine",
            "CountryComplianceChecker",
            "ThirdPartyAuditEngine",
            "ComplianceReportingEngine",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 10,
        "request_model_count": 12,
        "response_model_count": 12,
        "metrics_count": 20,
        "db_prefix": "gl_eudr_lcv_",
        "metrics_prefix": "gl_eudr_lcv_",
        "env_prefix": "GL_EUDR_LCV_",
        "scoring": {
            "compliant_threshold": 80,
            "partial_threshold": 50,
            "red_flag_critical": 75,
            "red_flag_high": 50,
            "red_flag_moderate": 25,
            "combined_risk_compliance_weight": 0.60,
            "combined_risk_red_flag_weight": 0.40,
        },
        "performance_targets": {
            "single_check_ms": 500,
            "full_assessment_ms": 5000,
        },
    }
