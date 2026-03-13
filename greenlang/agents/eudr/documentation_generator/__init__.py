# -*- coding: utf-8 -*-
"""
AGENT-EUDR-030: Documentation Generator Agent

EUDR Due Diligence Statement (DDS) generation, Article 9 data assembly,
risk assessment documentation, mitigation measure documentation,
compliance package building, document versioning, and regulatory
submission engine. Provides production-grade capabilities for generating
complete DDS packages that satisfy EU 2023/1115 Article 4(2) reporting
obligations, assembling Article 9 required data elements, documenting
risk assessments and mitigation measures with full provenance, building
compliance-ready submission packages, managing document versions with
audit trails, and submitting to the EU Information System.

The agent sits downstream of the Risk Assessment Engine (EUDR-028) and
the Mitigation Measure Designer (EUDR-029), consuming risk assessment
outputs and mitigation reports to produce complete Due Diligence
Statements with regulatory-compliant documentation packages for
submission to EU competent authorities.

Core capabilities:
    1. DDSStatementGenerator        -- Generates complete Due Diligence
       Statements per EUDR Article 4(2), assembling operator information,
       product details, geolocation references, supplier data, risk
       assessments, and mitigation summaries into a structured DDS
    2. Article9DataAssembler        -- Assembles the 10+ data elements
       required by EUDR Article 9 including product descriptions,
       quantities, country of production, geolocation coordinates,
       and supplier chain information
    3. RiskAssessmentDocumenter     -- Documents risk assessment results
       with composite scores, risk level classifications, contributing
       factor breakdowns, and regulatory cross-references for DDS
       inclusion
    4. MitigationDocumenter         -- Documents mitigation measures
       with before/after risk scores, measure summaries, effectiveness
       evidence, and Article 11 compliance status for DDS inclusion
    5. CompliancePackageBuilder     -- Builds complete compliance packages
       combining DDS, Article 9 data, risk documentation, mitigation
       documentation, and supporting evidence into a submission-ready
       bundle with integrity hashes
    6. DocumentVersionManager       -- Manages document version lifecycle
       with full audit trail, change tracking, version comparison, and
       retention policy enforcement per Article 12(3) five-year
       requirement
    7. RegulatorySubmissionEngine   -- Manages DDS submission to the EU
       Information System with status tracking, retry logic, receipt
       confirmation, and regulatory deadline monitoring

Foundational modules:
    - config.py       -- DocumentationGeneratorConfig with GL_EUDR_DGN_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 18 Prometheus self-monitoring metrics (gl_eudr_dgn_)

Agent ID: GL-EUDR-DGN-030
Module: greenlang.agents.eudr.documentation_generator
PRD: PRD-AGENT-EUDR-030
Regulation: EU 2023/1115 Articles 4, 9, 10, 11, 12, 13, 14-16, 29, 31

Example:
    >>> from greenlang.agents.eudr.documentation_generator import (
    ...     DocumentationGeneratorConfig,
    ...     get_config,
    ...     EUDRCommodity,
    ...     RiskLevel,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.retention_years)
    5

    >>> from greenlang.agents.eudr.documentation_generator import (
    ...     DocumentationGeneratorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-DGN-030"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "DocumentationGeneratorConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "EUDRCommodity",
    "RiskLevel",
    "DDSStatus",
    "DocumentType",
    "SubmissionStatus",
    "PackageFormat",
    "ValidationSeverity",
    "VersionStatus",
    "Article9Element",
    "ComplianceSection",
    "RetentionStatus",
    "AuditAction",
    # -- Core Models (15) --
    "DDSDocument",
    "Article9Package",
    "RiskAssessmentDoc",
    "MitigationDoc",
    "CompliancePackage",
    "DocumentVersion",
    "SubmissionRecord",
    "ValidationResult",
    "ValidationIssue",
    "ProductEntry",
    "GeolocationReference",
    "SupplierReference",
    "MeasureSummary",
    "DDSContent",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics --
    "record_dds_generated",
    "record_article9_assembled",
    "record_risk_documented",
    "record_mitigation_documented",
    "record_package_built",
    "record_submission_sent",
    "record_validation_run",
    "record_api_error",
    "observe_dds_generation_duration",
    "observe_article9_assembly_duration",
    "observe_package_build_duration",
    "observe_submission_duration",
    "set_active_dds_documents",
    "set_active_packages",
    "set_active_submissions",
    "set_pending_submissions",
    "set_validation_pass_rate",
    "set_total_versions",
    # -- Engines (7) --
    "DDSStatementGenerator",
    "Article9DataAssembler",
    "RiskAssessmentDocumenter",
    "MitigationDocumenter",
    "CompliancePackageBuilder",
    "DocumentVersionManager",
    "RegulatorySubmissionEngine",
    # -- Service Facade --
    "DocumentationGeneratorService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "DocumentationGeneratorConfig": ("config", "DocumentationGeneratorConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "EUDRCommodity": ("models", "EUDRCommodity"),
    "RiskLevel": ("models", "RiskLevel"),
    "DDSStatus": ("models", "DDSStatus"),
    "DocumentType": ("models", "DocumentType"),
    "SubmissionStatus": ("models", "SubmissionStatus"),
    "PackageFormat": ("models", "PackageFormat"),
    "ValidationSeverity": ("models", "ValidationSeverity"),
    "VersionStatus": ("models", "VersionStatus"),
    "Article9Element": ("models", "Article9Element"),
    "ComplianceSection": ("models", "ComplianceSection"),
    "RetentionStatus": ("models", "RetentionStatus"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15)
    "DDSDocument": ("models", "DDSDocument"),
    "Article9Package": ("models", "Article9Package"),
    "RiskAssessmentDoc": ("models", "RiskAssessmentDoc"),
    "MitigationDoc": ("models", "MitigationDoc"),
    "CompliancePackage": ("models", "CompliancePackage"),
    "DocumentVersion": ("models", "DocumentVersion"),
    "SubmissionRecord": ("models", "SubmissionRecord"),
    "ValidationResult": ("models", "ValidationResult"),
    "ValidationIssue": ("models", "ValidationIssue"),
    "ProductEntry": ("models", "ProductEntry"),
    "GeolocationReference": ("models", "GeolocationReference"),
    "SupplierReference": ("models", "SupplierReference"),
    "MeasureSummary": ("models", "MeasureSummary"),
    "DDSContent": ("models", "DDSContent"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_dds_generated": ("metrics", "record_dds_generated"),
    "record_article9_assembled": ("metrics", "record_article9_assembled"),
    "record_risk_documented": ("metrics", "record_risk_documented"),
    "record_mitigation_documented": (
        "metrics", "record_mitigation_documented",
    ),
    "record_package_built": ("metrics", "record_package_built"),
    "record_submission_sent": ("metrics", "record_submission_sent"),
    "record_validation_run": ("metrics", "record_validation_run"),
    "record_api_error": ("metrics", "record_api_error"),
    # Metrics (histograms)
    "observe_dds_generation_duration": (
        "metrics", "observe_dds_generation_duration",
    ),
    "observe_article9_assembly_duration": (
        "metrics", "observe_article9_assembly_duration",
    ),
    "observe_package_build_duration": (
        "metrics", "observe_package_build_duration",
    ),
    "observe_submission_duration": (
        "metrics", "observe_submission_duration",
    ),
    # Metrics (gauges)
    "set_active_dds_documents": ("metrics", "set_active_dds_documents"),
    "set_active_packages": ("metrics", "set_active_packages"),
    "set_active_submissions": ("metrics", "set_active_submissions"),
    "set_pending_submissions": ("metrics", "set_pending_submissions"),
    "set_validation_pass_rate": ("metrics", "set_validation_pass_rate"),
    "set_total_versions": ("metrics", "set_total_versions"),
    # Engines (7)
    "DDSStatementGenerator": (
        "dds_statement_generator", "DDSStatementGenerator",
    ),
    "Article9DataAssembler": (
        "article9_data_assembler", "Article9DataAssembler",
    ),
    "RiskAssessmentDocumenter": (
        "risk_assessment_documenter", "RiskAssessmentDocumenter",
    ),
    "MitigationDocumenter": (
        "mitigation_documenter", "MitigationDocumenter",
    ),
    "CompliancePackageBuilder": (
        "compliance_package_builder", "CompliancePackageBuilder",
    ),
    "DocumentVersionManager": (
        "document_version_manager", "DocumentVersionManager",
    ),
    "RegulatorySubmissionEngine": (
        "regulatory_submission_engine", "RegulatorySubmissionEngine",
    ),
    # Service Facade
    "DocumentationGeneratorService": (
        "setup", "DocumentationGeneratorService",
    ),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.documentation_generator import X``
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
            f"greenlang.agents.eudr.documentation_generator.{module_suffix}"
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
        engine listing, and capability summary for the Documentation
        Generator agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-DGN-030'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Documentation Generator",
        "prd": "PRD-AGENT-EUDR-030",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": [
            "4", "9", "10", "11", "12", "13", "14", "15", "16", "29", "31",
        ],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "document_types": [
            "due_diligence_statement",
            "article9_package",
            "risk_assessment_doc",
            "mitigation_doc",
            "compliance_package",
        ],
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        ],
        "article9_elements": [
            "product_description",
            "quantity",
            "country_of_production",
            "geolocation",
            "supplier_information",
            "date_of_production",
            "hs_code",
            "trade_name",
            "scientific_name",
            "operator_information",
        ],
        "engines": [
            "DDSStatementGenerator",
            "Article9DataAssembler",
            "RiskAssessmentDocumenter",
            "MitigationDocumenter",
            "CompliancePackageBuilder",
            "DocumentVersionManager",
            "RegulatorySubmissionEngine",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 18,
        "db_prefix": "gl_eudr_dgn_",
        "metrics_prefix": "gl_eudr_dgn_",
        "env_prefix": "GL_EUDR_DGN_",
    }
