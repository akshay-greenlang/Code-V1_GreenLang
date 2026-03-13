# -*- coding: utf-8 -*-
"""
AGENT-EUDR-035: Improvement Plan Creator

Creates comprehensive EUDR compliance improvement plans by aggregating
findings from upstream agents (EUDR-016 through EUDR-034), performing gap
analysis, generating SMART actions, mapping root causes via 5-Whys and
fishbone analysis, prioritizing with Eisenhower + risk-based scoring,
tracking progress with milestones and effectiveness reviews, and
coordinating stakeholders via RACI assignments and notifications.

Core capabilities:
    1. FindingAggregator       -- Collect, deduplicate, and classify
       findings from upstream EUDR monitoring agents
    2. GapAnalyzer             -- Map findings to EUDR regulatory gaps
       with severity scoring and article references
    3. ActionGenerator         -- Create SMART improvement actions with
       effort/cost estimation and deadline assignment
    4. RootCauseMapper         -- 5-Whys and fishbone (Ishikawa) root
       cause analysis with systemic cause identification
    5. PrioritizationEngine    -- Eisenhower matrix + multi-factor
       risk-based action prioritization
    6. ProgressTracker         -- Milestone tracking, progress snapshots,
       overdue detection, and effectiveness review
    7. StakeholderCoordinator  -- RACI assignments, multi-channel
       notifications, acknowledgment tracking, and escalation

Foundational modules:
    - config.py       -- ImprovementPlanCreatorConfig with GL_EUDR_IPC_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 12 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 42 Prometheus self-monitoring metrics (gl_eudr_ipc_)

Agent ID: GL-EUDR-IPC-035
Module: greenlang.agents.eudr.improvement_plan_creator
PRD: PRD-AGENT-EUDR-035
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 29, 31

Example:
    >>> from greenlang.agents.eudr.improvement_plan_creator import (
    ...     ImprovementPlanCreatorConfig,
    ...     get_config,
    ...     PlanStatus,
    ...     GapSeverity,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.max_actions_per_plan)
    50

    >>> from greenlang.agents.eudr.improvement_plan_creator import (
    ...     ImprovementPlanCreatorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-035 Improvement Plan Creator (GL-EUDR-IPC-035)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-IPC-035"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "ImprovementPlanCreatorConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12) --
    "EUDRCommodity",
    "RiskLevel",
    "GapSeverity",
    "ActionStatus",
    "ActionType",
    "PlanStatus",
    "EisenhowerQuadrant",
    "RACIRole",
    "FindingSource",
    "FishboneCategory",
    "NotificationChannel",
    # -- Core Models (15+) --
    "Finding",
    "AggregatedFindings",
    "ComplianceGap",
    "RootCause",
    "FishboneAnalysis",
    "ImprovementAction",
    "StakeholderAssignment",
    "ProgressMilestone",
    "ProgressSnapshot",
    "NotificationRecord",
    "ImprovementPlan",
    "PlanSummary",
    "PlanReport",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "DEFAULT_PRIORITY_WEIGHTS",
    "GAP_SEVERITY_THRESHOLDS",
    "SUPPORTED_COMMODITIES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (42) - Counters --
    "record_plan_created",
    "record_finding_aggregated",
    "record_gap_identified",
    "record_action_generated",
    "record_root_cause_mapped",
    "record_action_prioritized",
    "record_progress_snapshot",
    "record_stakeholder_assigned",
    "record_notification_sent",
    "record_action_completed",
    "record_action_verified",
    "record_escalation_triggered",
    "record_plan_approved",
    "record_report_generated",
    "record_duplicates_removed",
    "record_smart_validation",
    # -- Metrics (42) - Histograms --
    "observe_finding_aggregation_duration",
    "observe_gap_analysis_duration",
    "observe_action_generation_duration",
    "observe_root_cause_mapping_duration",
    "observe_prioritization_duration",
    "observe_progress_tracking_duration",
    "observe_stakeholder_coord_duration",
    "observe_plan_creation_duration",
    "observe_report_generation_duration",
    "observe_notification_dispatch_duration",
    "observe_fishbone_analysis_duration",
    "observe_five_whys_duration",
    "observe_effectiveness_review_duration",
    "observe_raci_validation_duration",
    # -- Metrics (42) - Gauges --
    "set_active_plans",
    "set_pending_actions",
    "set_overdue_actions",
    "set_critical_gaps_open",
    "set_high_gaps_open",
    "set_overall_progress",
    "set_avg_effectiveness",
    "set_stakeholders_pending_ack",
    "set_on_track_plans",
    "set_off_track_plans",
    "set_actions_on_hold",
    "set_systemic_root_causes",
    # -- Engines (7) --
    "FindingAggregator",
    "GapAnalyzer",
    "ActionGenerator",
    "RootCauseMapper",
    "PrioritizationEngine",
    "ProgressTracker",
    "StakeholderCoordinator",
    # -- Service Facade --
    "ImprovementPlanCreatorService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "ImprovementPlanCreatorConfig": ("config", "ImprovementPlanCreatorConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12)
    "EUDRCommodity": ("models", "EUDRCommodity"),
    "RiskLevel": ("models", "RiskLevel"),
    "GapSeverity": ("models", "GapSeverity"),
    "ActionStatus": ("models", "ActionStatus"),
    "ActionType": ("models", "ActionType"),
    "PlanStatus": ("models", "PlanStatus"),
    "EisenhowerQuadrant": ("models", "EisenhowerQuadrant"),
    "RACIRole": ("models", "RACIRole"),
    "FindingSource": ("models", "FindingSource"),
    "FishboneCategory": ("models", "FishboneCategory"),
    "NotificationChannel": ("models", "NotificationChannel"),
    # Core Models (15+)
    "Finding": ("models", "Finding"),
    "AggregatedFindings": ("models", "AggregatedFindings"),
    "ComplianceGap": ("models", "ComplianceGap"),
    "RootCause": ("models", "RootCause"),
    "FishboneAnalysis": ("models", "FishboneAnalysis"),
    "ImprovementAction": ("models", "ImprovementAction"),
    "StakeholderAssignment": ("models", "StakeholderAssignment"),
    "ProgressMilestone": ("models", "ProgressMilestone"),
    "ProgressSnapshot": ("models", "ProgressSnapshot"),
    "NotificationRecord": ("models", "NotificationRecord"),
    "ImprovementPlan": ("models", "ImprovementPlan"),
    "PlanSummary": ("models", "PlanSummary"),
    "PlanReport": ("models", "PlanReport"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "DEFAULT_PRIORITY_WEIGHTS": ("models", "DEFAULT_PRIORITY_WEIGHTS"),
    "GAP_SEVERITY_THRESHOLDS": ("models", "GAP_SEVERITY_THRESHOLDS"),
    "SUPPORTED_COMMODITIES": ("models", "SUPPORTED_COMMODITIES"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters - 16)
    "record_plan_created": ("metrics", "record_plan_created"),
    "record_finding_aggregated": ("metrics", "record_finding_aggregated"),
    "record_gap_identified": ("metrics", "record_gap_identified"),
    "record_action_generated": ("metrics", "record_action_generated"),
    "record_root_cause_mapped": ("metrics", "record_root_cause_mapped"),
    "record_action_prioritized": ("metrics", "record_action_prioritized"),
    "record_progress_snapshot": ("metrics", "record_progress_snapshot"),
    "record_stakeholder_assigned": ("metrics", "record_stakeholder_assigned"),
    "record_notification_sent": ("metrics", "record_notification_sent"),
    "record_action_completed": ("metrics", "record_action_completed"),
    "record_action_verified": ("metrics", "record_action_verified"),
    "record_escalation_triggered": ("metrics", "record_escalation_triggered"),
    "record_plan_approved": ("metrics", "record_plan_approved"),
    "record_report_generated": ("metrics", "record_report_generated"),
    "record_duplicates_removed": ("metrics", "record_duplicates_removed"),
    "record_smart_validation": ("metrics", "record_smart_validation"),
    # Metrics (histograms - 14)
    "observe_finding_aggregation_duration": ("metrics", "observe_finding_aggregation_duration"),
    "observe_gap_analysis_duration": ("metrics", "observe_gap_analysis_duration"),
    "observe_action_generation_duration": ("metrics", "observe_action_generation_duration"),
    "observe_root_cause_mapping_duration": ("metrics", "observe_root_cause_mapping_duration"),
    "observe_prioritization_duration": ("metrics", "observe_prioritization_duration"),
    "observe_progress_tracking_duration": ("metrics", "observe_progress_tracking_duration"),
    "observe_stakeholder_coord_duration": ("metrics", "observe_stakeholder_coord_duration"),
    "observe_plan_creation_duration": ("metrics", "observe_plan_creation_duration"),
    "observe_report_generation_duration": ("metrics", "observe_report_generation_duration"),
    "observe_notification_dispatch_duration": ("metrics", "observe_notification_dispatch_duration"),
    "observe_fishbone_analysis_duration": ("metrics", "observe_fishbone_analysis_duration"),
    "observe_five_whys_duration": ("metrics", "observe_five_whys_duration"),
    "observe_effectiveness_review_duration": ("metrics", "observe_effectiveness_review_duration"),
    "observe_raci_validation_duration": ("metrics", "observe_raci_validation_duration"),
    # Metrics (gauges - 12)
    "set_active_plans": ("metrics", "set_active_plans"),
    "set_pending_actions": ("metrics", "set_pending_actions"),
    "set_overdue_actions": ("metrics", "set_overdue_actions"),
    "set_critical_gaps_open": ("metrics", "set_critical_gaps_open"),
    "set_high_gaps_open": ("metrics", "set_high_gaps_open"),
    "set_overall_progress": ("metrics", "set_overall_progress"),
    "set_avg_effectiveness": ("metrics", "set_avg_effectiveness"),
    "set_stakeholders_pending_ack": ("metrics", "set_stakeholders_pending_ack"),
    "set_on_track_plans": ("metrics", "set_on_track_plans"),
    "set_off_track_plans": ("metrics", "set_off_track_plans"),
    "set_actions_on_hold": ("metrics", "set_actions_on_hold"),
    "set_systemic_root_causes": ("metrics", "set_systemic_root_causes"),
    # Engines (7)
    "FindingAggregator": ("finding_aggregator", "FindingAggregator"),
    "GapAnalyzer": ("gap_analyzer", "GapAnalyzer"),
    "ActionGenerator": ("action_generator", "ActionGenerator"),
    "RootCauseMapper": ("root_cause_mapper", "RootCauseMapper"),
    "PrioritizationEngine": ("prioritization_engine", "PrioritizationEngine"),
    "ProgressTracker": ("progress_tracker", "ProgressTracker"),
    "StakeholderCoordinator": ("stakeholder_coordinator", "StakeholderCoordinator"),
    # Service Facade
    "ImprovementPlanCreatorService": ("setup", "ImprovementPlanCreatorService"),
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
            f"greenlang.agents.eudr.improvement_plan_creator.{module_suffix}"
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
        "name": "Improvement Plan Creator",
        "prd": "PRD-AGENT-EUDR-035",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["10", "11", "12", "29", "31"],
        "upstream_dependencies": [
            "AGENT-EUDR-028 (Risk Assessment Engine)",
            "AGENT-EUDR-029 (Mitigation Measure Designer)",
            "AGENT-EUDR-030 (Documentation Generator)",
            "AGENT-EUDR-026 (Due Diligence Orchestrator)",
            "AGENT-EUDR-016 (Country Risk Evaluator)",
            "AGENT-EUDR-017 (Supplier Risk Scorer)",
            "AGENT-EUDR-034 (Annual Review Scheduler)",
        ],
        "engines": [
            "FindingAggregator",
            "GapAnalyzer",
            "ActionGenerator",
            "RootCauseMapper",
            "PrioritizationEngine",
            "ProgressTracker",
            "StakeholderCoordinator",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 42,
        "db_prefix": "gl_eudr_ipc_",
        "metrics_prefix": "gl_eudr_ipc_",
        "env_prefix": "GL_EUDR_IPC_",
    }
