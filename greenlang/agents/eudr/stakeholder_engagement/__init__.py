# -*- coding: utf-8 -*-
"""
AGENT-EUDR-031: Stakeholder Engagement Tool

EUDR stakeholder engagement lifecycle management including centralized
stakeholder registry, Free Prior and Informed Consent (FPIC) workflow
management, grievance mechanism operations, consultation record keeping,
multi-channel communication, indigenous rights engagement verification,
and audit-ready compliance reporting. Provides production-grade
capabilities for managing the full lifecycle of stakeholder engagement
across all stakeholder categories as required by EU 2023/1115 Articles
2, 4, 8, 9, 10, 11, 12, 29, 31 and aligned with ILO Convention 169,
UNDRIP, and CSDDD Articles 7-9.

The agent sits alongside the Indigenous Rights Checker (EUDR-021), the
Risk Mitigation Advisor (EUDR-025), and the Due Diligence Orchestrator
(EUDR-026), providing dedicated stakeholder engagement infrastructure
that integrates with supply chain traceability (EUDR-001), geolocation
verification (EUDR-002), and documentation generation (EUDR-030).

Core capabilities:
    1. StakeholderMapper              -- Centralized registry of all
       stakeholders across the EUDR supply chain, categorized by type,
       with rights classification, legal protections, engagement history,
       and supply chain node mapping
    2. FPICWorkflowEngine             -- Multi-stage workflow for managing
       Free, Prior and Informed Consent processes per ILO Convention 169
       and UNDRIP with 7 mandatory stages, configurable SLAs, evidence
       requirements, and approval gates
    3. GrievanceMechanism             -- Structured complaint management
       system compliant with UN Guiding Principles Principle 31
       effectiveness criteria with triage, investigation, resolution,
       appeal, and satisfaction assessment
    4. ConsultationRecordManager      -- Structured documentation system
       for all consultations with indigenous peoples, local communities,
       and other stakeholders per EUDR Article 10(2)(e) with immutable
       records and SHA-256 provenance
    5. CommunicationHub               -- Multi-channel stakeholder
       communication platform with template library, campaign management,
       delivery tracking, and 12+ language support
    6. IndigenousRightsEngagementVerifier -- Verification engine assessing
       engagement quality across 6 dimensions: cultural appropriateness,
       language accessibility, deliberation time, inclusiveness,
       genuineness, and decision respect
    7. ComplianceReporter             -- Audit-ready stakeholder
       engagement documentation generation for DDS submission,
       competent authority inspection, and certification scheme audits

Foundational modules:
    - config.py       -- StakeholderEngagementConfig with GL_EUDR_SET_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 18 Prometheus self-monitoring metrics (gl_eudr_set_)

Agent ID: GL-EUDR-SET-031
Module: greenlang.agents.eudr.stakeholder_engagement
PRD: PRD-AGENT-EUDR-031
Regulation: EU 2023/1115 Articles 2, 4, 8, 9, 10, 11, 12, 29, 31;
            ILO Convention 169; UNDRIP; CSDDD Articles 7, 8, 9

Example:
    >>> from greenlang.agents.eudr.stakeholder_engagement import (
    ...     StakeholderEngagementConfig,
    ...     get_config,
    ...     StakeholderType,
    ...     FPICStage,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.fpic_default_deliberation_days)
    90

    >>> from greenlang.agents.eudr.stakeholder_engagement import (
    ...     StakeholderEngagementService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-031 Stakeholder Engagement Tool (GL-EUDR-SET-031)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-SET-031"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "StakeholderEngagementConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "StakeholderType",
    "RelationshipStatus",
    "FPICStage",
    "ConsentStatus",
    "GrievanceSeverity",
    "GrievanceCategory",
    "GrievanceStatus",
    "IntakeChannel",
    "ConsultationType",
    "CommunicationChannel",
    "ReportType",
    "AuditAction",
    # -- Core Models (15+) --
    "StakeholderRecord",
    "FPICWorkflow",
    "GrievanceRecord",
    "ConsultationRecord",
    "CommunicationRecord",
    "EngagementAssessment",
    "ComplianceReport",
    "AuditEntry",
    "ContactInfo",
    "RightsClassification",
    "FPICStageConfig",
    "InvestigationNotes",
    "ResolutionActions",
    "FollowUpAction",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "SUPPORTED_LANGUAGES",
    "FPIC_MANDATORY_STAGES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics --
    "record_stakeholder_registered",
    "record_fpic_workflow_started",
    "record_fpic_consent_granted",
    "record_grievance_submitted",
    "record_grievance_resolved",
    "record_consultation_recorded",
    "record_communication_sent",
    "record_assessment_completed",
    "record_report_generated",
    "observe_fpic_workflow_duration",
    "observe_grievance_resolution_duration",
    "observe_consultation_duration",
    "observe_engagement_assessment_duration",
    "observe_report_generation_duration",
    "set_active_stakeholders",
    "set_active_fpic_workflows",
    "set_open_grievances",
    "set_pending_communications",
    # -- Engines (7) --
    "StakeholderMapper",
    "FPICWorkflowEngine",
    "GrievanceMechanism",
    "ConsultationRecordManager",
    "CommunicationHub",
    "IndigenousRightsEngagementVerifier",
    "ComplianceReporter",
    # -- Service Facade --
    "StakeholderEngagementService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "StakeholderEngagementConfig": ("config", "StakeholderEngagementConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "StakeholderType": ("models", "StakeholderType"),
    "RelationshipStatus": ("models", "RelationshipStatus"),
    "FPICStage": ("models", "FPICStage"),
    "ConsentStatus": ("models", "ConsentStatus"),
    "GrievanceSeverity": ("models", "GrievanceSeverity"),
    "GrievanceCategory": ("models", "GrievanceCategory"),
    "GrievanceStatus": ("models", "GrievanceStatus"),
    "IntakeChannel": ("models", "IntakeChannel"),
    "ConsultationType": ("models", "ConsultationType"),
    "CommunicationChannel": ("models", "CommunicationChannel"),
    "ReportType": ("models", "ReportType"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15+)
    "StakeholderRecord": ("models", "StakeholderRecord"),
    "FPICWorkflow": ("models", "FPICWorkflow"),
    "GrievanceRecord": ("models", "GrievanceRecord"),
    "ConsultationRecord": ("models", "ConsultationRecord"),
    "CommunicationRecord": ("models", "CommunicationRecord"),
    "EngagementAssessment": ("models", "EngagementAssessment"),
    "ComplianceReport": ("models", "ComplianceReport"),
    "AuditEntry": ("models", "AuditEntry"),
    "ContactInfo": ("models", "ContactInfo"),
    "RightsClassification": ("models", "RightsClassification"),
    "FPICStageConfig": ("models", "FPICStageConfig"),
    "InvestigationNotes": ("models", "InvestigationNotes"),
    "ResolutionActions": ("models", "ResolutionActions"),
    "FollowUpAction": ("models", "FollowUpAction"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "SUPPORTED_LANGUAGES": ("models", "SUPPORTED_LANGUAGES"),
    "FPIC_MANDATORY_STAGES": ("models", "FPIC_MANDATORY_STAGES"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_stakeholder_registered": (
        "metrics", "record_stakeholder_registered",
    ),
    "record_fpic_workflow_started": (
        "metrics", "record_fpic_workflow_started",
    ),
    "record_fpic_consent_granted": (
        "metrics", "record_fpic_consent_granted",
    ),
    "record_grievance_submitted": (
        "metrics", "record_grievance_submitted",
    ),
    "record_grievance_resolved": (
        "metrics", "record_grievance_resolved",
    ),
    "record_consultation_recorded": (
        "metrics", "record_consultation_recorded",
    ),
    "record_communication_sent": (
        "metrics", "record_communication_sent",
    ),
    "record_assessment_completed": (
        "metrics", "record_assessment_completed",
    ),
    "record_report_generated": (
        "metrics", "record_report_generated",
    ),
    # Metrics (histograms)
    "observe_fpic_workflow_duration": (
        "metrics", "observe_fpic_workflow_duration",
    ),
    "observe_grievance_resolution_duration": (
        "metrics", "observe_grievance_resolution_duration",
    ),
    "observe_consultation_duration": (
        "metrics", "observe_consultation_duration",
    ),
    "observe_engagement_assessment_duration": (
        "metrics", "observe_engagement_assessment_duration",
    ),
    "observe_report_generation_duration": (
        "metrics", "observe_report_generation_duration",
    ),
    # Metrics (gauges)
    "set_active_stakeholders": ("metrics", "set_active_stakeholders"),
    "set_active_fpic_workflows": ("metrics", "set_active_fpic_workflows"),
    "set_open_grievances": ("metrics", "set_open_grievances"),
    "set_pending_communications": ("metrics", "set_pending_communications"),
    # Engines (7)
    "StakeholderMapper": (
        "stakeholder_mapper", "StakeholderMapper",
    ),
    "FPICWorkflowEngine": (
        "fpic_workflow_engine", "FPICWorkflowEngine",
    ),
    "GrievanceMechanism": (
        "grievance_mechanism", "GrievanceMechanism",
    ),
    "ConsultationRecordManager": (
        "consultation_record_manager", "ConsultationRecordManager",
    ),
    "CommunicationHub": (
        "communication_hub", "CommunicationHub",
    ),
    "IndigenousRightsEngagementVerifier": (
        "engagement_verifier", "IndigenousRightsEngagementVerifier",
    ),
    "ComplianceReporter": (
        "compliance_reporter", "ComplianceReporter",
    ),
    # Service Facade
    "StakeholderEngagementService": (
        "setup", "StakeholderEngagementService",
    ),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.stakeholder_engagement import X``
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
            f"greenlang.agents.eudr.stakeholder_engagement.{module_suffix}"
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
        engine listing, and capability summary for the Stakeholder
        Engagement Tool agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-SET-031'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Stakeholder Engagement Tool",
        "prd": "PRD-AGENT-EUDR-031",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": [
            "2", "4", "8", "9", "10", "11", "12", "29", "31",
        ],
        "supplementary_frameworks": [
            "ILO Convention 169",
            "UN Declaration on the Rights of Indigenous Peoples (UNDRIP)",
            "EU Corporate Sustainability Due Diligence Directive (CSDDD)",
            "UN Guiding Principles on Business and Human Rights",
        ],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "stakeholder_types": [
            "indigenous_peoples",
            "local_communities",
            "smallholder_cooperatives",
            "ngos",
            "workers_unions",
            "government_authorities",
            "certification_bodies",
            "civil_society",
            "academic_institutions",
            "media",
        ],
        "fpic_stages": [
            "identification",
            "information_provision",
            "deliberation",
            "consultation",
            "consent_recording",
            "agreement_documentation",
            "ongoing_monitoring",
        ],
        "grievance_channels": [
            "web_portal", "mobile_app", "sms", "email",
            "community_submission",
        ],
        "supported_languages": [
            "en", "fr", "de", "es", "pt", "id", "sw",
            "ar", "zh", "hi", "ms", "th",
        ],
        "engines": [
            "StakeholderMapper",
            "FPICWorkflowEngine",
            "GrievanceMechanism",
            "ConsultationRecordManager",
            "CommunicationHub",
            "IndigenousRightsEngagementVerifier",
            "ComplianceReporter",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 18,
        "db_prefix": "gl_eudr_set_",
        "metrics_prefix": "gl_eudr_set_",
        "env_prefix": "GL_EUDR_SET_",
    }
