# -*- coding: utf-8 -*-
"""
AGENT-EUDR-037: Due Diligence Statement Creator

Creates, validates, and manages Due Diligence Statements (DDS) per EUDR
Article 4, including geolocation formatting per Article 9, risk data
integration from 10 upstream agents (EUDR-016 to 025), supply chain
compilation from 15 upstream agents (EUDR-001 to 015), compliance
validation against all 14 Article 4 mandatory fields, document packaging
with SHA-256 hashes, version control with amendment tracking, digital
signing per eIDAS Regulation, and EU Information System submission.

The agent supports all 24 EU official languages for DDS rendering and
maintains complete SHA-256 provenance chains for every operation per
Article 31 record-keeping requirements.

Core capabilities:
    1. StatementAssembler     -- Assembles DDS records with Article 4
       mandatory fields, auto-generates reference numbers (GL-DDS-
       YYYYMMDD-XXXX), manages lifecycle from draft to submission,
       and creates initial version records
    2. GeolocationFormatter   -- Formats geolocation data per Article 9
       with 4ha polygon threshold, WGS84 coordinate precision rounding,
       polygon closure enforcement, and GeoJSON export
    3. RiskDataIntegrator     -- Aggregates risk assessments from 10
       upstream EUDR agents into unified risk profiles with weighted
       scoring, risk level classification, and mitigation identification
    4. SupplyChainCompiler    -- Compiles supply chain data from 15
       upstream agents with tier tracking, traceability scoring,
       country aggregation, and completeness validation
    5. ComplianceValidator    -- Validates DDS against all 14 Article 4
       mandatory fields, geolocation completeness (Art 9), risk
       assessment inclusion (Art 10), supply chain data, and
       deforestation-free / legally-produced declarations
    6. DocumentPackager       -- Packages evidence documents (certificates,
       imagery, reports) with file validation, size limits, SHA-256
       hashing, and manifest generation for EU IS upload
    7. VersionController      -- Manages version history, amendments with
       reason tracking, digital signatures per eIDAS, signature
       validation, and statement withdrawal

Foundational modules:
    - config.py       -- DDSCreatorConfig with GL_EUDR_DDSC_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 40 Prometheus self-monitoring metrics (gl_eudr_ddsc_)

Agent ID: GL-EUDR-DDSC-037
Module: greenlang.agents.eudr.due_diligence_statement_creator
PRD: PRD-AGENT-EUDR-037
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 12, 13, 14, 31, 33
            eIDAS Regulation (EU) 910/2014

Example:
    >>> from greenlang.agents.eudr.due_diligence_statement_creator import (
    ...     DDSCreatorConfig,
    ...     get_config,
    ...     DDSStatus,
    ...     CommodityType,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.default_language)
    en

    >>> from greenlang.agents.eudr.due_diligence_statement_creator import (
    ...     DDSCreatorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-037 Due Diligence Statement Creator (GL-EUDR-DDSC-037)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-DDSC-037"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "DDSCreatorConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "DDSStatus",
    "CommodityType",
    "RiskLevel",
    "ComplianceStatus",
    "DocumentType",
    "SignatureType",
    "ValidationResult",
    "SubmissionStatus",
    "AmendmentReason",
    "GeolocationMethod",
    "LanguageCode",
    "StatementType",
    # -- Core Models (15+) --
    "DDSStatement",
    "GeolocationData",
    "RiskReference",
    "SupplyChainData",
    "ComplianceCheck",
    "DocumentPackage",
    "StatementVersion",
    "DigitalSignature",
    "DDSValidationReport",
    "SubmissionPackage",
    "AmendmentRecord",
    "TemplateConfig",
    "LanguageTranslation",
    "StatementSummary",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EUDR_REGULATED_COMMODITIES",
    "ARTICLE_4_MANDATORY_FIELDS",
    "EU_OFFICIAL_LANGUAGES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (40 helpers) --
    "record_statement_created",
    "record_statement_submitted",
    "record_amendment_created",
    "record_validation_passed",
    "record_validation_failed",
    "record_document_packaged",
    "record_signature_applied",
    "record_geolocation_formatted",
    "record_risk_integration",
    "record_supply_chain_compilation",
    "record_version_created",
    "record_withdrawal",
    "record_translation",
    "record_batch_operation",
    "observe_statement_generation_duration",
    "observe_validation_duration",
    "observe_geolocation_formatting_duration",
    "observe_risk_integration_duration",
    "observe_supply_chain_compilation_duration",
    "observe_document_packaging_duration",
    "observe_signing_duration",
    "observe_submission_duration",
    "observe_amendment_duration",
    "observe_translation_duration",
    "observe_version_creation_duration",
    "set_active_statements",
    "set_pending_submissions",
    "set_failed_validations",
    "set_total_commodity_volume",
    "set_draft_statements",
    "set_validated_statements",
    "set_signed_statements",
    "set_submitted_statements",
    "set_accepted_statements",
    "set_rejected_statements",
    "set_amended_statements",
    "set_withdrawn_statements",
    "set_total_documents",
    "set_total_geolocations",
    "set_average_risk_score",
    # -- Engines (7) --
    "StatementAssembler",
    "GeolocationFormatter",
    "RiskDataIntegrator",
    "SupplyChainCompiler",
    "ComplianceValidator",
    "DocumentPackager",
    "VersionController",
    # -- Service Facade --
    "DDSCreatorService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "DDSCreatorConfig": ("config", "DDSCreatorConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "DDSStatus": ("models", "DDSStatus"),
    "CommodityType": ("models", "CommodityType"),
    "RiskLevel": ("models", "RiskLevel"),
    "ComplianceStatus": ("models", "ComplianceStatus"),
    "DocumentType": ("models", "DocumentType"),
    "SignatureType": ("models", "SignatureType"),
    "ValidationResult": ("models", "ValidationResult"),
    "SubmissionStatus": ("models", "SubmissionStatus"),
    "AmendmentReason": ("models", "AmendmentReason"),
    "GeolocationMethod": ("models", "GeolocationMethod"),
    "LanguageCode": ("models", "LanguageCode"),
    "StatementType": ("models", "StatementType"),
    # Core Models (15+)
    "DDSStatement": ("models", "DDSStatement"),
    "GeolocationData": ("models", "GeolocationData"),
    "RiskReference": ("models", "RiskReference"),
    "SupplyChainData": ("models", "SupplyChainData"),
    "ComplianceCheck": ("models", "ComplianceCheck"),
    "DocumentPackage": ("models", "DocumentPackage"),
    "StatementVersion": ("models", "StatementVersion"),
    "DigitalSignature": ("models", "DigitalSignature"),
    "DDSValidationReport": ("models", "DDSValidationReport"),
    "SubmissionPackage": ("models", "SubmissionPackage"),
    "AmendmentRecord": ("models", "AmendmentRecord"),
    "TemplateConfig": ("models", "TemplateConfig"),
    "LanguageTranslation": ("models", "LanguageTranslation"),
    "StatementSummary": ("models", "StatementSummary"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "EUDR_REGULATED_COMMODITIES": ("models", "EUDR_REGULATED_COMMODITIES"),
    "ARTICLE_4_MANDATORY_FIELDS": ("models", "ARTICLE_4_MANDATORY_FIELDS"),
    "EU_OFFICIAL_LANGUAGES": ("models", "EU_OFFICIAL_LANGUAGES"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics - Counters (14)
    "record_statement_created": (
        "metrics", "record_statement_created",
    ),
    "record_statement_submitted": (
        "metrics", "record_statement_submitted",
    ),
    "record_amendment_created": (
        "metrics", "record_amendment_created",
    ),
    "record_validation_passed": (
        "metrics", "record_validation_passed",
    ),
    "record_validation_failed": (
        "metrics", "record_validation_failed",
    ),
    "record_document_packaged": (
        "metrics", "record_document_packaged",
    ),
    "record_signature_applied": (
        "metrics", "record_signature_applied",
    ),
    "record_geolocation_formatted": (
        "metrics", "record_geolocation_formatted",
    ),
    "record_risk_integration": (
        "metrics", "record_risk_integration",
    ),
    "record_supply_chain_compilation": (
        "metrics", "record_supply_chain_compilation",
    ),
    "record_version_created": (
        "metrics", "record_version_created",
    ),
    "record_withdrawal": (
        "metrics", "record_withdrawal",
    ),
    "record_translation": (
        "metrics", "record_translation",
    ),
    "record_batch_operation": (
        "metrics", "record_batch_operation",
    ),
    # Metrics - Histograms (11)
    "observe_statement_generation_duration": (
        "metrics", "observe_statement_generation_duration",
    ),
    "observe_validation_duration": (
        "metrics", "observe_validation_duration",
    ),
    "observe_geolocation_formatting_duration": (
        "metrics", "observe_geolocation_formatting_duration",
    ),
    "observe_risk_integration_duration": (
        "metrics", "observe_risk_integration_duration",
    ),
    "observe_supply_chain_compilation_duration": (
        "metrics", "observe_supply_chain_compilation_duration",
    ),
    "observe_document_packaging_duration": (
        "metrics", "observe_document_packaging_duration",
    ),
    "observe_signing_duration": (
        "metrics", "observe_signing_duration",
    ),
    "observe_submission_duration": (
        "metrics", "observe_submission_duration",
    ),
    "observe_amendment_duration": (
        "metrics", "observe_amendment_duration",
    ),
    "observe_translation_duration": (
        "metrics", "observe_translation_duration",
    ),
    "observe_version_creation_duration": (
        "metrics", "observe_version_creation_duration",
    ),
    # Metrics - Gauges (15)
    "set_active_statements": ("metrics", "set_active_statements"),
    "set_pending_submissions": ("metrics", "set_pending_submissions"),
    "set_failed_validations": ("metrics", "set_failed_validations"),
    "set_total_commodity_volume": ("metrics", "set_total_commodity_volume"),
    "set_draft_statements": ("metrics", "set_draft_statements"),
    "set_validated_statements": ("metrics", "set_validated_statements"),
    "set_signed_statements": ("metrics", "set_signed_statements"),
    "set_submitted_statements": ("metrics", "set_submitted_statements"),
    "set_accepted_statements": ("metrics", "set_accepted_statements"),
    "set_rejected_statements": ("metrics", "set_rejected_statements"),
    "set_amended_statements": ("metrics", "set_amended_statements"),
    "set_withdrawn_statements": ("metrics", "set_withdrawn_statements"),
    "set_total_documents": ("metrics", "set_total_documents"),
    "set_total_geolocations": ("metrics", "set_total_geolocations"),
    "set_average_risk_score": ("metrics", "set_average_risk_score"),
    # Engines (7)
    "StatementAssembler": (
        "statement_assembler", "StatementAssembler",
    ),
    "GeolocationFormatter": (
        "geolocation_formatter", "GeolocationFormatter",
    ),
    "RiskDataIntegrator": (
        "risk_data_integrator", "RiskDataIntegrator",
    ),
    "SupplyChainCompiler": (
        "supply_chain_compiler", "SupplyChainCompiler",
    ),
    "ComplianceValidator": (
        "compliance_validator", "ComplianceValidator",
    ),
    "DocumentPackager": (
        "document_packager", "DocumentPackager",
    ),
    "VersionController": (
        "version_controller", "VersionController",
    ),
    # Service Facade
    "DDSCreatorService": ("setup", "DDSCreatorService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.due_diligence_statement_creator import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: If the name is not a known export.
    """
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.due_diligence_statement_creator.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        engine listing, and capability summary for the Due Diligence
        Statement Creator agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-DDSC-037'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Due Diligence Statement Creator",
        "prd": "PRD-AGENT-EUDR-037",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": [
            "4", "8", "9", "10", "12", "13", "14", "31", "33",
        ],
        "supplementary_frameworks": [
            "eIDAS Regulation (EU) 910/2014",
        ],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
        "article_4_mandatory_fields": [
            "operator_name",
            "operator_address",
            "operator_eori_number",
            "commodity_type",
            "product_description",
            "hs_code",
            "country_of_production",
            "geolocation_of_plots",
            "quantity",
            "supplier_information",
            "compliance_declaration",
            "risk_assessment_outcome",
            "risk_mitigation_measures",
            "date_of_statement",
        ],
        "eu_official_languages": [
            "bg", "cs", "da", "de", "el", "en", "es", "et",
            "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
            "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
        ],
        "upstream_risk_agents": [
            "EUDR-016", "EUDR-017", "EUDR-018", "EUDR-019",
            "EUDR-020", "EUDR-021", "EUDR-022", "EUDR-023",
            "EUDR-024", "EUDR-025",
        ],
        "upstream_supply_chain_agents": [
            f"EUDR-{i:03d}" for i in range(1, 16)
        ],
        "engines": [
            "StatementAssembler",
            "GeolocationFormatter",
            "RiskDataIntegrator",
            "SupplyChainCompiler",
            "ComplianceValidator",
            "DocumentPackager",
            "VersionController",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 40,
        "db_prefix": "gl_eudr_ddsc_",
        "metrics_prefix": "gl_eudr_ddsc_",
        "env_prefix": "GL_EUDR_DDSC_",
    }
