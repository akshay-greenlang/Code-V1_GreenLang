# -*- coding: utf-8 -*-
"""
AGENT-EUDR-032: Grievance Mechanism Manager

Advanced grievance analytics, mediation, remediation tracking, risk scoring,
collective grievance handling, and regulatory reporting built on top of the
basic grievance mechanism provided by EUDR-031 (Stakeholder Engagement Tool).
This agent reads from EUDR-031's gl_eudr_set_grievances table but adds its
own analytics, mediation, remediation, risk, collective grievance, and
regulatory reporting capabilities.

Core capabilities:
    1. GrievanceAnalyticsEngine        -- Pattern detection, trend analysis,
       clustering of grievances across operators, time periods, and categories
    2. RootCauseAnalyzer               -- Deterministic root cause analysis
       using five-whys, fishbone, fault-tree, and correlation methods
    3. MediationWorkflowManager        -- Multi-party mediation state machine
       with 7-stage workflow: initiated -> preparation -> dialogue ->
       negotiation -> settlement -> implementation -> closed
    4. RemediationTracker              -- Effectiveness measurement,
       stakeholder satisfaction, cost tracking, timeline adherence, and
       verification evidence collection
    5. RiskScoringEngine               -- Predictive risk analytics across
       operator, supplier, commodity, and region scopes with historical
       trend analysis and confidence scoring
    6. CollectiveGrievanceHandler      -- Class-action/collective complaint
       management with demand tracking, negotiation workflow, and
       representative body coordination
    7. RegulatoryReporter              -- EUDR Article 16, CSDDD Article 8,
       UNGP effectiveness, and annual summary report generation

Foundational modules:
    - config.py       -- GrievanceMechanismManagerConfig with GL_EUDR_GMM_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 18 Prometheus self-monitoring metrics (gl_eudr_gmm_)

Agent ID: GL-EUDR-GMM-032
Module: greenlang.agents.eudr.grievance_mechanism_manager
PRD: PRD-AGENT-EUDR-032
Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31;
            CSDDD Article 8; UNGP Principle 31

Example:
    >>> from greenlang.agents.eudr.grievance_mechanism_manager import (
    ...     GrievanceMechanismManagerConfig,
    ...     get_config,
    ...     PatternType,
    ...     MediationStage,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.mediation_max_sessions)
    20

    >>> from greenlang.agents.eudr.grievance_mechanism_manager import (
    ...     GrievanceMechanismManagerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-032 Grievance Mechanism Manager (GL-EUDR-GMM-032)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-GMM-032"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "GrievanceMechanismManagerConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "PatternType",
    "TrendDirection",
    "AnalysisMethod",
    "MediationStage",
    "MediatorType",
    "SettlementStatus",
    "RemediationType",
    "ImplementationStatus",
    "RiskScope",
    "RiskLevel",
    "CollectiveStatus",
    "NegotiationStatus",
    "RegulatoryReportType",
    "AuditAction",
    # -- Core Models (15+) --
    "GrievanceAnalyticsRecord",
    "RootCauseRecord",
    "MediationRecord",
    "RemediationRecord",
    "RiskScoreRecord",
    "CollectiveGrievanceRecord",
    "RegulatoryReport",
    "AuditEntry",
    "CausalChainStep",
    "MediationSession",
    "RemediationAction",
    "CollectiveDemand",
    "ScoreFactor",
    "ReportSection",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "MEDIATION_STAGES_ORDERED",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (18) --
    "record_analytics_created",
    "record_root_cause_analyzed",
    "record_mediation_initiated",
    "record_mediation_completed",
    "record_remediation_created",
    "record_remediation_verified",
    "record_risk_score_computed",
    "record_collective_created",
    "record_regulatory_report_generated",
    "observe_analytics_duration",
    "observe_root_cause_duration",
    "observe_mediation_session_duration",
    "observe_remediation_verification_duration",
    "observe_risk_scoring_duration",
    "observe_report_generation_duration",
    "set_active_mediations",
    "set_open_remediations",
    "set_high_risk_entities",
    # -- Engines (7) --
    "GrievanceAnalyticsEngine",
    "RootCauseAnalyzer",
    "MediationWorkflowManager",
    "RemediationTracker",
    "RiskScoringEngine",
    "CollectiveGrievanceHandler",
    "RegulatoryReporter",
    # -- Service Facade --
    "GrievanceMechanismManagerService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "GrievanceMechanismManagerConfig": ("config", "GrievanceMechanismManagerConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "PatternType": ("models", "PatternType"),
    "TrendDirection": ("models", "TrendDirection"),
    "AnalysisMethod": ("models", "AnalysisMethod"),
    "MediationStage": ("models", "MediationStage"),
    "MediatorType": ("models", "MediatorType"),
    "SettlementStatus": ("models", "SettlementStatus"),
    "RemediationType": ("models", "RemediationType"),
    "ImplementationStatus": ("models", "ImplementationStatus"),
    "RiskScope": ("models", "RiskScope"),
    "RiskLevel": ("models", "RiskLevel"),
    "CollectiveStatus": ("models", "CollectiveStatus"),
    "NegotiationStatus": ("models", "NegotiationStatus"),
    "RegulatoryReportType": ("models", "RegulatoryReportType"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15+)
    "GrievanceAnalyticsRecord": ("models", "GrievanceAnalyticsRecord"),
    "RootCauseRecord": ("models", "RootCauseRecord"),
    "MediationRecord": ("models", "MediationRecord"),
    "RemediationRecord": ("models", "RemediationRecord"),
    "RiskScoreRecord": ("models", "RiskScoreRecord"),
    "CollectiveGrievanceRecord": ("models", "CollectiveGrievanceRecord"),
    "RegulatoryReport": ("models", "RegulatoryReport"),
    "AuditEntry": ("models", "AuditEntry"),
    "CausalChainStep": ("models", "CausalChainStep"),
    "MediationSession": ("models", "MediationSession"),
    "RemediationAction": ("models", "RemediationAction"),
    "CollectiveDemand": ("models", "CollectiveDemand"),
    "ScoreFactor": ("models", "ScoreFactor"),
    "ReportSection": ("models", "ReportSection"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "MEDIATION_STAGES_ORDERED": ("models", "MEDIATION_STAGES_ORDERED"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_analytics_created": ("metrics", "record_analytics_created"),
    "record_root_cause_analyzed": ("metrics", "record_root_cause_analyzed"),
    "record_mediation_initiated": ("metrics", "record_mediation_initiated"),
    "record_mediation_completed": ("metrics", "record_mediation_completed"),
    "record_remediation_created": ("metrics", "record_remediation_created"),
    "record_remediation_verified": ("metrics", "record_remediation_verified"),
    "record_risk_score_computed": ("metrics", "record_risk_score_computed"),
    "record_collective_created": ("metrics", "record_collective_created"),
    "record_regulatory_report_generated": ("metrics", "record_regulatory_report_generated"),
    # Metrics (histograms)
    "observe_analytics_duration": ("metrics", "observe_analytics_duration"),
    "observe_root_cause_duration": ("metrics", "observe_root_cause_duration"),
    "observe_mediation_session_duration": ("metrics", "observe_mediation_session_duration"),
    "observe_remediation_verification_duration": ("metrics", "observe_remediation_verification_duration"),
    "observe_risk_scoring_duration": ("metrics", "observe_risk_scoring_duration"),
    "observe_report_generation_duration": ("metrics", "observe_report_generation_duration"),
    # Metrics (gauges)
    "set_active_mediations": ("metrics", "set_active_mediations"),
    "set_open_remediations": ("metrics", "set_open_remediations"),
    "set_high_risk_entities": ("metrics", "set_high_risk_entities"),
    # Engines (7)
    "GrievanceAnalyticsEngine": ("grievance_analytics_engine", "GrievanceAnalyticsEngine"),
    "RootCauseAnalyzer": ("root_cause_analyzer", "RootCauseAnalyzer"),
    "MediationWorkflowManager": ("mediation_workflow_manager", "MediationWorkflowManager"),
    "RemediationTracker": ("remediation_tracker", "RemediationTracker"),
    "RiskScoringEngine": ("risk_scoring_engine", "RiskScoringEngine"),
    "CollectiveGrievanceHandler": ("collective_grievance_handler", "CollectiveGrievanceHandler"),
    "RegulatoryReporter": ("regulatory_reporter", "RegulatoryReporter"),
    # Service Facade
    "GrievanceMechanismManagerService": ("setup", "GrievanceMechanismManagerService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports."""
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.grievance_mechanism_manager.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string."""
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata."""
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Grievance Mechanism Manager",
        "prd": "PRD-AGENT-EUDR-032",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["2", "4", "8", "9", "10", "11", "12", "29", "31"],
        "supplementary_frameworks": [
            "EU Corporate Sustainability Due Diligence Directive (CSDDD) Article 8",
            "UN Guiding Principles on Business and Human Rights Principle 31",
            "ILO Convention 169",
        ],
        "upstream_dependency": "AGENT-EUDR-031 (Stakeholder Engagement Tool)",
        "engines": [
            "GrievanceAnalyticsEngine",
            "RootCauseAnalyzer",
            "MediationWorkflowManager",
            "RemediationTracker",
            "RiskScoringEngine",
            "CollectiveGrievanceHandler",
            "RegulatoryReporter",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 18,
        "db_prefix": "gl_eudr_gmm_",
        "metrics_prefix": "gl_eudr_gmm_",
        "env_prefix": "GL_EUDR_GMM_",
    }
