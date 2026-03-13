# -*- coding: utf-8 -*-
"""
AGENT-EUDR-021: Indigenous Rights Checker (GL-EUDR-IRC-021)

Production-grade indigenous rights checking agent for EUDR compliance.
Provides territory database management, FPIC verification with 10-element
weighted scoring, PostGIS-based land rights overlap detection, community
consultation lifecycle tracking, rights violation monitoring and severity
scoring, indigenous community registry, FPIC workflow management, and
compliance reporting in multiple formats and languages.

Regulation: EU 2023/1115 (EUDR) - Articles 2, 8, 10, 11, 29, 31
Zero-Hallucination: All numeric scoring uses deterministic Decimal arithmetic.
Provenance: SHA-256 chain-hashed audit trail for every operation.

Engines (7):
    1. TerritoryDatabaseEngine      - Feature 1: Indigenous territory management
    2. FPICVerificationEngine        - Feature 2: FPIC documentation verification
    3. LandRightsOverlapEngine       - Feature 3: PostGIS overlap detection
    4. CommunityConsultationEngine   - Feature 4: Community engagement tracking
    5. RightsViolationEngine         - Feature 5: Violation monitoring & scoring
    6. IndigenousRegistryEngine      - Feature 6: Community database
    7. ComplianceReportingEngine     - Feature 8: Report generation

Reference Data (3):
    - ilo_169_countries: 24 ILO Convention 169 ratifying countries
    - indigenous_territory_sources: 6 authoritative data sources
    - fpic_legal_frameworks: 8 country FPIC legal frameworks

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker import (
    ...     get_service,
    ...     get_config,
    ...     IndigenousRightsCheckerService,
    ...     IndigenousRightsCheckerConfig,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version and agent identification
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-IRC-021"
__agent_name__ = "Indigenous Rights Checker"

# ---------------------------------------------------------------------------
# Configuration imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.config import (
        IndigenousRightsCheckerConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError as e:
    logger.warning(f"Could not import config: {e}")
    IndigenousRightsCheckerConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.models import (
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_BATCH_SIZE,
        SRID_WGS84,
        EUDR_COMMODITIES,
        SUPPORTED_REGIONS,
        # Enumerations
        TerritoryLegalStatus,
        FPICStatus,
        OverlapType,
        ViolationType,
        FPICWorkflowStage,
        ConsultationStage,
        GrievanceStatus,
        AgreementStatus,
        ConfidenceLevel,
        RiskLevel,
        AlertSeverity,
        DataSource,
        ReportType,
        ReportFormat,
        CommunityRecognitionStatus,
        CountryRiskLevel,
        ViolationAlertStatus,
        # Core models
        IndigenousTerritory,
        FPICAssessment,
        TerritoryOverlap,
        IndigenousCommunity,
        ConsultationRecord,
        GrievanceRecord,
        BenefitSharingAgreement,
        FPICWorkflow,
        WorkflowTransition,
        ViolationAlert,
        ComplianceReport,
        CountryIndigenousRightsScore,
        AuditLogEntry,
        # Request models
        DetectOverlapRequest,
        BatchOverlapRequest,
        VerifyFPICRequest,
        CreateWorkflowRequest,
        AdvanceWorkflowRequest,
        GenerateReportRequest,
        CorrelateViolationsRequest,
        RecordConsultationRequest,
        SubmitGrievanceRequest,
        HealthCheckRequest,
        # Response models
        OverlapDetectionResponse,
        BatchOverlapResponse,
        FPICVerificationResponse,
        WorkflowStatusResponse,
        ViolationCorrelationResponse,
        HealthCheckResponse,
    )
except ImportError as e:
    logger.warning(f"Could not import models: {e}")

# ---------------------------------------------------------------------------
# Provenance imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        get_tracker,
        reset_tracker,
    )
except ImportError as e:
    logger.warning(f"Could not import provenance: {e}")
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    get_tracker = None  # type: ignore[assignment]
    reset_tracker = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Metrics imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
        PROMETHEUS_AVAILABLE,
        record_territory_query,
        record_fpic_assessment,
        record_overlap_detected,
        record_consultation_recorded,
        record_violation_ingested,
        record_violation_correlated,
        record_workflow_created,
        record_workflow_transition,
        record_report_generated,
        record_api_error,
        observe_overlap_query_duration,
        observe_fpic_assessment_duration,
        set_active_territories,
        set_active_overlaps,
        set_active_workflows,
    )
except ImportError as e:
    logger.warning(f"Could not import metrics: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Engine imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.territory_database_engine import (
        TerritoryDatabaseEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import TerritoryDatabaseEngine: {e}")
    TerritoryDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.fpic_verification_engine import (
        FPICVerificationEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import FPICVerificationEngine: {e}")
    FPICVerificationEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.land_rights_overlap_engine import (
        LandRightsOverlapEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import LandRightsOverlapEngine: {e}")
    LandRightsOverlapEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.community_consultation_engine import (
        CommunityConsultationEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import CommunityConsultationEngine: {e}")
    CommunityConsultationEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.rights_violation_engine import (
        RightsViolationEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import RightsViolationEngine: {e}")
    RightsViolationEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.indigenous_registry_engine import (
        IndigenousRegistryEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import IndigenousRegistryEngine: {e}")
    IndigenousRegistryEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.indigenous_rights_checker.compliance_reporting_engine import (
        ComplianceReportingEngine,
    )
except ImportError as e:
    logger.warning(f"Could not import ComplianceReportingEngine: {e}")
    ComplianceReportingEngine = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Setup / Service facade imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.setup import (
        IndigenousRightsCheckerService,
        get_service,
        reset_service,
        lifespan,
    )
except ImportError as e:
    logger.warning(f"Could not import setup: {e}")
    IndigenousRightsCheckerService = None  # type: ignore[assignment,misc]
    get_service = None  # type: ignore[assignment]
    reset_service = None  # type: ignore[assignment]
    lifespan = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Reference data imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.reference_data.ilo_169_countries import (
        ILO_169_COUNTRIES,
        is_ilo_169_ratified,
        get_ilo_169_data,
        get_eudr_relevant_ratifiers,
    )
except ImportError as e:
    logger.warning(f"Could not import ILO 169 reference data: {e}")

try:
    from greenlang.agents.eudr.indigenous_rights_checker.reference_data.indigenous_territory_sources import (
        TERRITORY_DATA_SOURCES,
        get_source_config,
        get_sources_for_country,
        get_total_estimated_territories,
    )
except ImportError as e:
    logger.warning(f"Could not import territory sources reference data: {e}")

try:
    from greenlang.agents.eudr.indigenous_rights_checker.reference_data.fpic_legal_frameworks import (
        FPIC_LEGAL_FRAMEWORKS,
        get_fpic_requirements,
        is_fpic_legally_required,
        get_consultation_protocol,
        get_minimum_consultation_days,
        get_countries_with_fpic_requirement,
    )
except ImportError as e:
    logger.warning(f"Could not import FPIC frameworks reference data: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Helper functions
    "get_version",
    "get_agent_info",
    # Configuration
    "IndigenousRightsCheckerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "TerritoryLegalStatus",
    "FPICStatus",
    "OverlapType",
    "ViolationType",
    "FPICWorkflowStage",
    "ConsultationStage",
    "GrievanceStatus",
    "AgreementStatus",
    "ConfidenceLevel",
    "RiskLevel",
    "AlertSeverity",
    "DataSource",
    "ReportType",
    "ReportFormat",
    "CommunityRecognitionStatus",
    "CountryRiskLevel",
    "ViolationAlertStatus",
    # Core models
    "IndigenousTerritory",
    "FPICAssessment",
    "TerritoryOverlap",
    "IndigenousCommunity",
    "ConsultationRecord",
    "GrievanceRecord",
    "BenefitSharingAgreement",
    "FPICWorkflow",
    "WorkflowTransition",
    "ViolationAlert",
    "ComplianceReport",
    "CountryIndigenousRightsScore",
    "AuditLogEntry",
    # Request models
    "DetectOverlapRequest",
    "BatchOverlapRequest",
    "VerifyFPICRequest",
    "CreateWorkflowRequest",
    "AdvanceWorkflowRequest",
    "GenerateReportRequest",
    "CorrelateViolationsRequest",
    "RecordConsultationRequest",
    "SubmitGrievanceRequest",
    "HealthCheckRequest",
    # Response models
    "OverlapDetectionResponse",
    "BatchOverlapResponse",
    "FPICVerificationResponse",
    "WorkflowStatusResponse",
    "ViolationCorrelationResponse",
    "HealthCheckResponse",
    # Constants
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "SRID_WGS84",
    "EUDR_COMMODITIES",
    "SUPPORTED_REGIONS",
    # Provenance
    "ProvenanceRecord",
    "ProvenanceTracker",
    "get_tracker",
    "reset_tracker",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "record_territory_query",
    "record_fpic_assessment",
    "record_overlap_detected",
    "record_consultation_recorded",
    "record_violation_ingested",
    "record_violation_correlated",
    "record_workflow_created",
    "record_workflow_transition",
    "record_report_generated",
    "record_api_error",
    "observe_overlap_query_duration",
    "observe_fpic_assessment_duration",
    "set_active_territories",
    "set_active_overlaps",
    "set_active_workflows",
    # Engines
    "TerritoryDatabaseEngine",
    "FPICVerificationEngine",
    "LandRightsOverlapEngine",
    "CommunityConsultationEngine",
    "RightsViolationEngine",
    "IndigenousRegistryEngine",
    "ComplianceReportingEngine",
    # Service facade
    "IndigenousRightsCheckerService",
    "get_service",
    "reset_service",
    "lifespan",
    # Reference data
    "ILO_169_COUNTRIES",
    "is_ilo_169_ratified",
    "get_ilo_169_data",
    "get_eudr_relevant_ratifiers",
    "TERRITORY_DATA_SOURCES",
    "get_source_config",
    "get_sources_for_country",
    "get_total_estimated_territories",
    "FPIC_LEGAL_FRAMEWORKS",
    "get_fpic_requirements",
    "is_fpic_legally_required",
    "get_consultation_protocol",
    "get_minimum_consultation_days",
    "get_countries_with_fpic_requirement",
]


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the agent module version string.

    Returns:
        Version string (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> Dict[str, Any]:
    """Return agent identification and capability summary.

    Returns:
        Dictionary with agent_id, version, name, engine count,
        engine names, and feature list.

    Example:
        >>> info = get_agent_info()
        >>> assert info["agent_id"] == "GL-EUDR-IRC-021"
        >>> assert len(info["engines"]) == 7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": __agent_name__,
        "regulation": "EU 2023/1115 (EUDR)",
        "applicable_articles": [
            "Article 2 (Indigenous Rights)",
            "Article 8 (Due Diligence)",
            "Article 10 (Risk Assessment)",
            "Article 11 (Risk Mitigation)",
            "Article 29 (Country Benchmarking)",
            "Article 31 (Record Keeping)",
        ],
        "engine_count": 7,
        "engines": [
            "TerritoryDatabaseEngine",
            "FPICVerificationEngine",
            "LandRightsOverlapEngine",
            "CommunityConsultationEngine",
            "RightsViolationEngine",
            "IndigenousRegistryEngine",
            "ComplianceReportingEngine",
        ],
        "features": [
            "F1: Indigenous Territory Database",
            "F2: FPIC Documentation Verification",
            "F3: Land Rights Overlap Detection",
            "F4: Community Consultation Tracker",
            "F5: Rights Violation Monitoring",
            "F6: Indigenous Community Registry",
            "F7: FPIC Workflow Management",
            "F8: Compliance Reporting",
            "F9: Country-Level Assessment",
        ],
        "reference_data": [
            "ILO 169 Countries (24)",
            "Territory Data Sources (6)",
            "FPIC Legal Frameworks (8)",
        ],
        "report_formats": ["pdf", "json", "html", "csv", "xlsx"],
        "report_languages": ["en", "fr", "de", "es", "pt"],
        "metrics_prefix": "gl_eudr_irc_",
        "metrics_count": 15,
        "zero_hallucination": True,
        "provenance_algorithm": "SHA-256",
    }
