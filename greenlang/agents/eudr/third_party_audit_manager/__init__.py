# -*- coding: utf-8 -*-
"""
Third-Party Audit Manager Agent - AGENT-EUDR-024

Production-grade third-party audit orchestration system for EUDR compliance
verification. Manages the full audit lifecycle from risk-based planning and
scheduling through auditor assignment, execution, non-conformance detection,
corrective action management, certification scheme integration, ISO 19011
report generation, competent authority liaison, and analytics dashboards.

The agent implements ISO 19011:2018 audit management principles and
ISO/IEC 17065:2012 conformity assessment requirements for five major
certification schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC) with
full EUDR Article mapping and zero-hallucination deterministic processing.

This package provides a complete third-party audit management system for
EUDR regulatory compliance per EU 2023/1115 Articles 3, 4, 9-11, 14-16,
18-23, 29, and 31:

    Capabilities:
        - Risk-based audit planning with composite priority scoring using
          weighted formula: Country_Risk*0.25 + Supplier_Risk*0.25 +
          NC_History*0.20 + Cert_Gap*0.15 + Deforestation_Alert*0.15,
          multiplied by Recency_Multiplier (capped at 2.0)
        - Auditor registry with ISO/IEC 17065/17021-1 competence tracking,
          conflict-of-interest detection (24-month cooling-off), CPD
          compliance monitoring, and multi-dimensional smart matching
        - Audit execution with scheme-specific checklists (EUDR 17 criteria,
          FSC 12, PEFC 8, RSPO 8, RA 7, ISCC 8), evidence collection with
          SHA-256 integrity, and ISO 19011 Annex A sampling plans using
          Cochran formula with finite population correction
        - Rule-based non-conformance detection with 20 classification rules
          (7 critical, 8 major, 5 minor) and root cause analysis using
          5-Whys and Ishikawa fishbone frameworks
        - CAR lifecycle management with 12 statuses, severity-based SLA
          deadlines (critical 30d, major 90d, minor 365d), and 4-level
          escalation (75%, 90%, SLA exceeded, SLA+30 days)
        - Certification scheme integration for FSC/PEFC/RSPO/RA/ISCC with
          EUDR coverage matrix (FSC 75%, PEFC 70%, RSPO 65%, RA 60%,
          ISCC 55%) and gap analysis
        - ISO 19011:2018 Clause 6.6 compliant report generation in 5
          formats (PDF/JSON/HTML/XLSX/XML) and 5 languages (EN/FR/DE/ES/PT)
        - Competent authority liaison for 27 EU Member States with SLA
          tracking and interaction logging
        - Analytics dashboards with finding trends, CAR performance,
          compliance rates, auditor metrics, and cost analysis

    Foundational modules:
        - config: ThirdPartyAuditManagerConfig with GL_EUDR_TAM_ env var
          support (55+ settings covering planning, auditor, execution,
          NC detection, CAR management, certification, reporting,
          authority liaison, and analytics)
        - models: Pydantic v2 data models with 7 enumerations, 10 core
          models, 7 request models, and 7 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions for full traceability
        - metrics: 20 Prometheus self-monitoring metrics (gl_eudr_tam_)
          covering audit scheduling, NC detection, CAR management,
          report generation, authority interaction, and analytics

PRD: PRD-AGENT-EUDR-024
Agent ID: GL-EUDR-TAM-024
Regulation: EU 2023/1115 (EUDR) Articles 3, 4, 9-11, 14-16, 18-23, 29, 31
ISO Standards: ISO 19011:2018, ISO/IEC 17065:2012, ISO/IEC 17021-1:2015
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.third_party_audit_manager import (
    ...     AuditStatus,
    ...     NCSeverity,
    ...     CARStatus,
    ...     ThirdPartyAuditManagerConfig,
    ...     get_config,
    ... )
    >>> from decimal import Decimal
    >>> cfg = get_config()
    >>> print(cfg.critical_sla_days, cfg.country_risk_weight)
    30 0.25

    >>> from greenlang.agents.eudr.third_party_audit_manager import (
    ...     Audit,
    ...     NonConformance,
    ...     CorrectiveActionRequest,
    ... )
    >>> audit = Audit(
    ...     audit_id="AUD-2026-001",
    ...     operator_id="OP-001",
    ...     operator_name="Timber Corp",
    ...     scope=AuditScope.EUDR_COMPLIANCE,
    ... )
    >>> print(audit.audit_id)
    'AUD-2026-001'

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-TAM-024"

# ---------------------------------------------------------------------------
# Foundational imports: config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.third_party_audit_manager.config import (
        ThirdPartyAuditManagerConfig,
        get_config,
        reset_config,
        set_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import config module: {e}")
    ThirdPartyAuditManagerConfig = None  # type: ignore[misc,assignment]
    get_config = None  # type: ignore[misc,assignment]
    reset_config = None  # type: ignore[misc,assignment]
    set_config = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: models
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.third_party_audit_manager.models import (
        # Enumerations (7)
        AuditStatus,
        AuditScope,
        AuditModality,
        NCSeverity,
        CARStatus,
        CertificationScheme,
        AuthorityInteractionType,
        # Core Models (10)
        Audit,
        Auditor,
        AuditChecklist,
        AuditEvidence,
        RootCauseAnalysis,
        NonConformance,
        CorrectiveActionRequest,
        CertificateRecord,
        CompetentAuthorityInteraction,
        AuditReport,
        # Request Models (7)
        ScheduleAuditRequest,
        MatchAuditorRequest,
        ClassifyNCRequest,
        IssueCARRequest,
        GenerateReportRequest,
        LogAuthorityInteractionRequest,
        CalculateAnalyticsRequest,
        # Response Models (7)
        ScheduleAuditResponse,
        MatchAuditorResponse,
        ClassifyNCResponse,
        IssueCARResponse,
        GenerateReportResponse,
        LogAuthorityInteractionResponse,
        CalculateAnalyticsResponse,
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_SCHEMES,
        SUPPORTED_COMMODITIES,
        SUPPORTED_REPORT_FORMATS,
        SUPPORTED_REPORT_LANGUAGES,
        NC_SEVERITY_SLA_DAYS,
        EU_MEMBER_STATES,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import models module: {e}")
    # Enumerations (7)
    AuditStatus = None  # type: ignore[misc,assignment]
    AuditScope = None  # type: ignore[misc,assignment]
    AuditModality = None  # type: ignore[misc,assignment]
    NCSeverity = None  # type: ignore[misc,assignment]
    CARStatus = None  # type: ignore[misc,assignment]
    CertificationScheme = None  # type: ignore[misc,assignment]
    AuthorityInteractionType = None  # type: ignore[misc,assignment]
    # Core Models (10)
    Audit = None  # type: ignore[misc,assignment]
    Auditor = None  # type: ignore[misc,assignment]
    AuditChecklist = None  # type: ignore[misc,assignment]
    AuditEvidence = None  # type: ignore[misc,assignment]
    RootCauseAnalysis = None  # type: ignore[misc,assignment]
    NonConformance = None  # type: ignore[misc,assignment]
    CorrectiveActionRequest = None  # type: ignore[misc,assignment]
    CertificateRecord = None  # type: ignore[misc,assignment]
    CompetentAuthorityInteraction = None  # type: ignore[misc,assignment]
    AuditReport = None  # type: ignore[misc,assignment]
    # Request Models (7)
    ScheduleAuditRequest = None  # type: ignore[misc,assignment]
    MatchAuditorRequest = None  # type: ignore[misc,assignment]
    ClassifyNCRequest = None  # type: ignore[misc,assignment]
    IssueCARRequest = None  # type: ignore[misc,assignment]
    GenerateReportRequest = None  # type: ignore[misc,assignment]
    LogAuthorityInteractionRequest = None  # type: ignore[misc,assignment]
    CalculateAnalyticsRequest = None  # type: ignore[misc,assignment]
    # Response Models (7)
    ScheduleAuditResponse = None  # type: ignore[misc,assignment]
    MatchAuditorResponse = None  # type: ignore[misc,assignment]
    ClassifyNCResponse = None  # type: ignore[misc,assignment]
    IssueCARResponse = None  # type: ignore[misc,assignment]
    GenerateReportResponse = None  # type: ignore[misc,assignment]
    LogAuthorityInteractionResponse = None  # type: ignore[misc,assignment]
    CalculateAnalyticsResponse = None  # type: ignore[misc,assignment]
    # Constants
    VERSION = None  # type: ignore[misc,assignment]
    EUDR_CUTOFF_DATE = None  # type: ignore[misc,assignment]
    MAX_BATCH_SIZE = None  # type: ignore[misc,assignment]
    EUDR_RETENTION_YEARS = None  # type: ignore[misc,assignment]
    SUPPORTED_SCHEMES = None  # type: ignore[misc,assignment]
    SUPPORTED_COMMODITIES = None  # type: ignore[misc,assignment]
    SUPPORTED_REPORT_FORMATS = None  # type: ignore[misc,assignment]
    SUPPORTED_REPORT_LANGUAGES = None  # type: ignore[misc,assignment]
    NC_SEVERITY_SLA_DAYS = None  # type: ignore[misc,assignment]
    EU_MEMBER_STATES = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.third_party_audit_manager.provenance import (
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
    from greenlang.agents.eudr.third_party_audit_manager.metrics import (
        PROMETHEUS_AVAILABLE,
        # Counter helpers (10)
        record_audit_scheduled,
        record_audit_completed,
        record_nc_detected,
        record_car_issued,
        record_car_closed,
        record_report_generated,
        record_authority_interaction,
        record_auditor_match,
        record_cert_sync,
        record_api_error,
        # Histogram helpers (4)
        observe_scheduling_duration,
        observe_nc_classification_duration,
        observe_report_generation_duration,
        observe_analytics_calculation_duration,
        # Gauge helpers (6)
        set_active_audits,
        set_open_cars,
        set_overdue_cars,
        set_pending_authority_responses,
        set_registered_auditors,
        set_active_certificates,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import metrics module: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    # Counter helpers (10)
    record_audit_scheduled = None  # type: ignore[misc,assignment]
    record_audit_completed = None  # type: ignore[misc,assignment]
    record_nc_detected = None  # type: ignore[misc,assignment]
    record_car_issued = None  # type: ignore[misc,assignment]
    record_car_closed = None  # type: ignore[misc,assignment]
    record_report_generated = None  # type: ignore[misc,assignment]
    record_authority_interaction = None  # type: ignore[misc,assignment]
    record_auditor_match = None  # type: ignore[misc,assignment]
    record_cert_sync = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]
    # Histogram helpers (4)
    observe_scheduling_duration = None  # type: ignore[misc,assignment]
    observe_nc_classification_duration = None  # type: ignore[misc,assignment]
    observe_report_generation_duration = None  # type: ignore[misc,assignment]
    observe_analytics_calculation_duration = None  # type: ignore[misc,assignment]
    # Gauge helpers (6)
    set_active_audits = None  # type: ignore[misc,assignment]
    set_open_cars = None  # type: ignore[misc,assignment]
    set_overdue_cars = None  # type: ignore[misc,assignment]
    set_pending_authority_responses = None  # type: ignore[misc,assignment]
    set_registered_auditors = None  # type: ignore[misc,assignment]
    set_active_certificates = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: Audit Planning and Scheduling ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
        AuditPlanningSchedulingEngine,
    )
except ImportError:
    AuditPlanningSchedulingEngine = None  # type: ignore[misc,assignment]

# ---- Engine 2: Auditor Registry and Qualification ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.auditor_registry_qualification_engine import (
        AuditorRegistryQualificationEngine,
    )
except ImportError:
    AuditorRegistryQualificationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 3: Audit Execution ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
        AuditExecutionEngine,
    )
except ImportError:
    AuditExecutionEngine = None  # type: ignore[misc,assignment]

# ---- Engine 4: Non-Conformance Detection ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
        NonConformanceDetectionEngine,
    )
except ImportError:
    NonConformanceDetectionEngine = None  # type: ignore[misc,assignment]

# ---- Engine 5: CAR Management ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
        CARManagementEngine,
    )
except ImportError:
    CARManagementEngine = None  # type: ignore[misc,assignment]

# ---- Engine 6: Certification Integration ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.certification_integration_engine import (
        CertificationIntegrationEngine,
    )
except ImportError:
    CertificationIntegrationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Audit Reporting ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.audit_reporting_engine import (
        AuditReportingEngine,
    )
except ImportError:
    AuditReportingEngine = None  # type: ignore[misc,assignment]

# ---- Engine 8: Audit Analytics (combined F8 Authority Liaison + F9 Analytics) ----
try:
    from greenlang.agents.eudr.third_party_audit_manager.audit_analytics_engine import (
        AuditAnalyticsEngine,
    )
except ImportError:
    AuditAnalyticsEngine = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Service facade import (conditional - setup may not exist yet)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.third_party_audit_manager.setup import (
        ThirdPartyAuditManagerSetup,
        get_setup,
        reset_setup,
    )
except ImportError:
    ThirdPartyAuditManagerSetup = None  # type: ignore[misc,assignment]
    get_setup = None  # type: ignore[misc,assignment]
    reset_setup = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Config --
    "ThirdPartyAuditManagerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Enumerations (7) --
    "AuditStatus",
    "AuditScope",
    "AuditModality",
    "NCSeverity",
    "CARStatus",
    "CertificationScheme",
    "AuthorityInteractionType",
    # -- Core Models (10) --
    "Audit",
    "Auditor",
    "AuditChecklist",
    "AuditEvidence",
    "RootCauseAnalysis",
    "NonConformance",
    "CorrectiveActionRequest",
    "CertificateRecord",
    "CompetentAuthorityInteraction",
    "AuditReport",
    # -- Request Models (7) --
    "ScheduleAuditRequest",
    "MatchAuditorRequest",
    "ClassifyNCRequest",
    "IssueCARRequest",
    "GenerateReportRequest",
    "LogAuthorityInteractionRequest",
    "CalculateAnalyticsRequest",
    # -- Response Models (7) --
    "ScheduleAuditResponse",
    "MatchAuditorResponse",
    "ClassifyNCResponse",
    "IssueCARResponse",
    "GenerateReportResponse",
    "LogAuthorityInteractionResponse",
    "CalculateAnalyticsResponse",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_SCHEMES",
    "SUPPORTED_COMMODITIES",
    "SUPPORTED_REPORT_FORMATS",
    "SUPPORTED_REPORT_LANGUAGES",
    "NC_SEVERITY_SLA_DAYS",
    "EU_MEMBER_STATES",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_tracker",
    "reset_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_audit_scheduled",
    "record_audit_completed",
    "record_nc_detected",
    "record_car_issued",
    "record_car_closed",
    "record_report_generated",
    "record_authority_interaction",
    "record_auditor_match",
    "record_cert_sync",
    "record_api_error",
    "observe_scheduling_duration",
    "observe_nc_classification_duration",
    "observe_report_generation_duration",
    "observe_analytics_calculation_duration",
    "set_active_audits",
    "set_open_cars",
    "set_overdue_cars",
    "set_pending_authority_responses",
    "set_registered_auditors",
    "set_active_certificates",
    # -- Engines (8) --
    "AuditPlanningSchedulingEngine",
    "AuditorRegistryQualificationEngine",
    "AuditExecutionEngine",
    "NonConformanceDetectionEngine",
    "CARManagementEngine",
    "CertificationIntegrationEngine",
    "AuditReportingEngine",
    "AuditAnalyticsEngine",
    # -- Setup Facade --
    "ThirdPartyAuditManagerSetup",
    "get_setup",
    "reset_setup",
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
        engine listing, certification schemes, and model counts for
        the Third-Party Audit Manager agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-TAM-024'
        >>> info["engine_count"]
        8
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Third-Party Audit Manager",
        "prd": "PRD-AGENT-EUDR-024",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": [
            "3", "4", "9", "10", "11", "14", "15", "16",
            "18", "19", "20", "21", "22", "23", "29", "31",
        ],
        "iso_standards": [
            "ISO 19011:2018 (Guidelines for auditing management systems)",
            "ISO/IEC 17065:2012 (Conformity assessment - Certification of products)",
            "ISO/IEC 17021-1:2015 (Conformity assessment - Audit and certification)",
        ],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "certification_schemes": [
            "FSC (Forest Stewardship Council, 5-year cycle)",
            "PEFC (Programme for Endorsement of Forest Certification, 5-year cycle)",
            "RSPO (Roundtable on Sustainable Palm Oil, 5-year cycle)",
            "Rainforest Alliance (3-year cycle)",
            "ISCC (International Sustainability and Carbon Certification, annual)",
        ],
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        ],
        "cutoff_date": "2020-12-31",
        "features": [
            "F1: Audit Planning and Scheduling",
            "F2: Auditor Registry and Qualification",
            "F3: Audit Execution and Monitoring",
            "F4: Non-Conformance Detection and Classification",
            "F5: Corrective Action Request Management",
            "F6: Certification Scheme Integration",
            "F7: ISO 19011 Report Generation",
            "F8: Competent Authority Liaison",
            "F9: Audit Analytics and Dashboards",
        ],
        "engines": [
            "AuditPlanningSchedulingEngine",
            "AuditorRegistryQualificationEngine",
            "AuditExecutionEngine",
            "NonConformanceDetectionEngine",
            "CARManagementEngine",
            "CertificationIntegrationEngine",
            "AuditReportingEngine",
            "AuditAnalyticsEngine",
        ],
        "engine_count": 8,
        "enum_count": 7,
        "core_model_count": 10,
        "request_model_count": 7,
        "response_model_count": 7,
        "nc_classification_rules": 20,
        "eu_checklist_criteria": 17,
        "eu_member_states": 27,
        "metrics_count": 20,
        "database_tables": 11,
        "report_formats": ["PDF", "JSON", "HTML", "XLSX", "XML"],
        "report_languages": ["EN", "FR", "DE", "ES", "PT"],
        "db_prefix": "gl_eudr_tam_",
        "metrics_prefix": "gl_eudr_tam_",
        "env_prefix": "GL_EUDR_TAM_",
    }
