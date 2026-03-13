# -*- coding: utf-8 -*-
"""
AGENT-EUDR-034: Annual Review Scheduler Agent

Manages annual EUDR compliance review cycles, regulatory deadline
tracking, commodity-specific checklist generation, multi-entity review
coordination, year-over-year data comparison, unified compliance
calendar management, and automated notification dispatch.

Core capabilities:
    1. ReviewCycleManager       -- Create, schedule, and manage annual
       review cycles with task generation and status tracking
    2. DeadlineTracker          -- Track regulatory and submission
       deadlines, manage submissions to competent authorities
    3. ChecklistGenerator       -- Generate commodity-specific review
       checklists from EUDR article requirements and templates
    4. EntityCoordinator        -- Coordinate reviews across
       organizational entities with dependency resolution
    5. YearComparator           -- Multi-year data comparison for
       trend identification and regression detection
    6. CalendarManager          -- Unified compliance calendar with
       iCal export and external calendar synchronization
    7. NotificationEngine       -- Automated multi-channel notifications
       with acknowledgment tracking and escalation tiers

Foundational modules:
    - config.py       -- AnnualReviewSchedulerConfig with GL_EUDR_ARS_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 14 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 40 Prometheus self-monitoring metrics (gl_eudr_ars_)

Agent ID: GL-EUDR-ARS-034
Module: greenlang.agents.eudr.annual_review_scheduler
PRD: PRD-AGENT-EUDR-034
Regulation: EU 2023/1115 (EUDR) Articles 8, 10, 11, 12, 14, 29, 31

Example:
    >>> from greenlang.agents.eudr.annual_review_scheduler import (
    ...     AnnualReviewSchedulerConfig,
    ...     get_config,
    ...     ReviewCycleStatus,
    ...     DeadlineStatus,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.review_cycle_duration_days)
    60

    >>> from greenlang.agents.eudr.annual_review_scheduler import (
    ...     AnnualReviewSchedulerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-034 Annual Review Scheduler Agent (GL-EUDR-ARS-034)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-ARS-034"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "AnnualReviewSchedulerConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (14) --
    "ReviewCycleStatus",
    "DeadlineStatus",
    "DeadlineType",
    "ChecklistItemStatus",
    "ChecklistPriority",
    "EntityType",
    "EntityReviewStatus",
    "ChangeSignificance",
    "ChangeDirection",
    "CalendarEventType",
    "NotificationChannel",
    "NotificationStatus",
    "EscalationLevel",
    "AuditAction",
    # -- Core Models (15+) --
    "ReviewCycleRecord",
    "DeadlineTrackingRecord",
    "ChecklistRecord",
    "EntityCoordinationRecord",
    "YearComparisonRecord",
    "CalendarRecord",
    "NotificationBatchRecord",
    "ReviewSummary",
    "AuditEntry",
    "HealthStatus",
    # -- Sub-models --
    "ReviewTask",
    "ChecklistItem",
    "DeadlineEntry",
    "EntityReviewInfo",
    "YearDataPoint",
    "YearComparison",
    "CalendarEvent",
    "NotificationRecord",
    "ActionRecommendation",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EUDR_ARTICLES_APPLICABLE",
    "EUDR_COMMODITIES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (40) --
    "record_review_cycle_created",
    "record_review_task_scheduled",
    "record_deadline_registered",
    "record_deadline_met",
    "record_deadline_overdue",
    "record_submission",
    "record_checklist_generated",
    "record_checklist_item_completed",
    "record_entity_coordinated",
    "record_entity_cascade",
    "record_year_comparison",
    "record_calendar_event_created",
    "record_notification_sent",
    "record_notification_acknowledged",
    "record_escalation_triggered",
    "record_review_completed",
    "observe_cycle_creation_duration",
    "observe_task_scheduling_duration",
    "observe_deadline_check_duration",
    "observe_submission_duration",
    "observe_checklist_generation_duration",
    "observe_entity_coordination_duration",
    "observe_cascade_resolution_duration",
    "observe_year_comparison_duration",
    "observe_comparison_report_duration",
    "observe_calendar_sync_duration",
    "observe_ical_generation_duration",
    "observe_notification_dispatch_duration",
    "observe_escalation_duration",
    "observe_summary_generation_duration",
    "set_active_review_cycles",
    "set_pending_tasks",
    "set_approaching_deadlines",
    "set_overdue_deadlines",
    "set_checklist_completion",
    "set_entity_completion",
    "set_pending_notifications",
    "set_upcoming_calendar_events",
    "set_overall_review_progress",
    "set_critical_changes_detected",
    # -- Engines (7) --
    "ReviewCycleManager",
    "DeadlineTracker",
    "ChecklistGenerator",
    "EntityCoordinator",
    "YearComparator",
    "CalendarManager",
    "NotificationEngine",
    # -- Service Facade --
    "AnnualReviewSchedulerService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "AnnualReviewSchedulerConfig": ("config", "AnnualReviewSchedulerConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (14)
    "ReviewCycleStatus": ("models", "ReviewCycleStatus"),
    "DeadlineStatus": ("models", "DeadlineStatus"),
    "DeadlineType": ("models", "DeadlineType"),
    "ChecklistItemStatus": ("models", "ChecklistItemStatus"),
    "ChecklistPriority": ("models", "ChecklistPriority"),
    "EntityType": ("models", "EntityType"),
    "EntityReviewStatus": ("models", "EntityReviewStatus"),
    "ChangeSignificance": ("models", "ChangeSignificance"),
    "ChangeDirection": ("models", "ChangeDirection"),
    "CalendarEventType": ("models", "CalendarEventType"),
    "NotificationChannel": ("models", "NotificationChannel"),
    "NotificationStatus": ("models", "NotificationStatus"),
    "EscalationLevel": ("models", "EscalationLevel"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15+)
    "ReviewCycleRecord": ("models", "ReviewCycleRecord"),
    "DeadlineTrackingRecord": ("models", "DeadlineTrackingRecord"),
    "ChecklistRecord": ("models", "ChecklistRecord"),
    "EntityCoordinationRecord": ("models", "EntityCoordinationRecord"),
    "YearComparisonRecord": ("models", "YearComparisonRecord"),
    "CalendarRecord": ("models", "CalendarRecord"),
    "NotificationBatchRecord": ("models", "NotificationBatchRecord"),
    "ReviewSummary": ("models", "ReviewSummary"),
    "AuditEntry": ("models", "AuditEntry"),
    "HealthStatus": ("models", "HealthStatus"),
    # Sub-models
    "ReviewTask": ("models", "ReviewTask"),
    "ChecklistItem": ("models", "ChecklistItem"),
    "DeadlineEntry": ("models", "DeadlineEntry"),
    "EntityReviewInfo": ("models", "EntityReviewInfo"),
    "YearDataPoint": ("models", "YearDataPoint"),
    "YearComparison": ("models", "YearComparison"),
    "CalendarEvent": ("models", "CalendarEvent"),
    "NotificationRecord": ("models", "NotificationRecord"),
    "ActionRecommendation": ("models", "ActionRecommendation"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "EUDR_ARTICLES_APPLICABLE": ("models", "EUDR_ARTICLES_APPLICABLE"),
    "EUDR_COMMODITIES": ("models", "EUDR_COMMODITIES"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters - 16)
    "record_review_cycle_created": ("metrics", "record_review_cycle_created"),
    "record_review_task_scheduled": ("metrics", "record_review_task_scheduled"),
    "record_deadline_registered": ("metrics", "record_deadline_registered"),
    "record_deadline_met": ("metrics", "record_deadline_met"),
    "record_deadline_overdue": ("metrics", "record_deadline_overdue"),
    "record_submission": ("metrics", "record_submission"),
    "record_checklist_generated": ("metrics", "record_checklist_generated"),
    "record_checklist_item_completed": ("metrics", "record_checklist_item_completed"),
    "record_entity_coordinated": ("metrics", "record_entity_coordinated"),
    "record_entity_cascade": ("metrics", "record_entity_cascade"),
    "record_year_comparison": ("metrics", "record_year_comparison"),
    "record_calendar_event_created": ("metrics", "record_calendar_event_created"),
    "record_notification_sent": ("metrics", "record_notification_sent"),
    "record_notification_acknowledged": ("metrics", "record_notification_acknowledged"),
    "record_escalation_triggered": ("metrics", "record_escalation_triggered"),
    "record_review_completed": ("metrics", "record_review_completed"),
    # Metrics (histograms - 14)
    "observe_cycle_creation_duration": ("metrics", "observe_cycle_creation_duration"),
    "observe_task_scheduling_duration": ("metrics", "observe_task_scheduling_duration"),
    "observe_deadline_check_duration": ("metrics", "observe_deadline_check_duration"),
    "observe_submission_duration": ("metrics", "observe_submission_duration"),
    "observe_checklist_generation_duration": ("metrics", "observe_checklist_generation_duration"),
    "observe_entity_coordination_duration": ("metrics", "observe_entity_coordination_duration"),
    "observe_cascade_resolution_duration": ("metrics", "observe_cascade_resolution_duration"),
    "observe_year_comparison_duration": ("metrics", "observe_year_comparison_duration"),
    "observe_comparison_report_duration": ("metrics", "observe_comparison_report_duration"),
    "observe_calendar_sync_duration": ("metrics", "observe_calendar_sync_duration"),
    "observe_ical_generation_duration": ("metrics", "observe_ical_generation_duration"),
    "observe_notification_dispatch_duration": ("metrics", "observe_notification_dispatch_duration"),
    "observe_escalation_duration": ("metrics", "observe_escalation_duration"),
    "observe_summary_generation_duration": ("metrics", "observe_summary_generation_duration"),
    # Metrics (gauges - 10)
    "set_active_review_cycles": ("metrics", "set_active_review_cycles"),
    "set_pending_tasks": ("metrics", "set_pending_tasks"),
    "set_approaching_deadlines": ("metrics", "set_approaching_deadlines"),
    "set_overdue_deadlines": ("metrics", "set_overdue_deadlines"),
    "set_checklist_completion": ("metrics", "set_checklist_completion"),
    "set_entity_completion": ("metrics", "set_entity_completion"),
    "set_pending_notifications": ("metrics", "set_pending_notifications"),
    "set_upcoming_calendar_events": ("metrics", "set_upcoming_calendar_events"),
    "set_overall_review_progress": ("metrics", "set_overall_review_progress"),
    "set_critical_changes_detected": ("metrics", "set_critical_changes_detected"),
    # Engine-specific Enums
    "EUDRCommodity": ("models", "EUDRCommodity"),
    "ReviewType": ("models", "ReviewType"),
    "ReviewPhase": ("models", "ReviewPhase"),
    "DeadlineAlertLevel": ("models", "DeadlineAlertLevel"),
    "EntityRole": ("models", "EntityRole"),
    "EntityStatus": ("models", "EntityStatus"),
    "CalendarEntryType": ("models", "CalendarEntryType"),
    "NotificationPriority": ("models", "NotificationPriority"),
    "YearComparisonStatus": ("models", "YearComparisonStatus"),
    "ComparisonDimension": ("models", "ComparisonDimension"),
    # Engine-specific Models
    "CommodityScope": ("models", "CommodityScope"),
    "ReviewPhaseConfig": ("models", "ReviewPhaseConfig"),
    "ReviewCycle": ("models", "ReviewCycle"),
    "DeadlineTrack": ("models", "DeadlineTrack"),
    "DeadlineAlert": ("models", "DeadlineAlert"),
    "ChecklistTemplate": ("models", "ChecklistTemplate"),
    "EntityCoordination": ("models", "EntityCoordination"),
    "EntityDependency": ("models", "EntityDependency"),
    "YearMetricSnapshot": ("models", "YearMetricSnapshot"),
    "ComparisonMetric": ("models", "ComparisonMetric"),
    "ComparisonResult": ("models", "ComparisonResult"),
    "CalendarEntry": ("models", "CalendarEntry"),
    "NotificationTemplate": ("models", "NotificationTemplate"),
    "YearDimensionComparison": ("models", "YearDimensionComparison"),
    # Engine-specific Constants
    "SUPPORTED_COMMODITIES": ("models", "SUPPORTED_COMMODITIES"),
    "REVIEW_PHASES_ORDER": ("models", "REVIEW_PHASES_ORDER"),
    # Engines (7)
    "ReviewCycleManager": ("review_cycle_manager", "ReviewCycleManager"),
    "DeadlineTracker": ("deadline_tracker", "DeadlineTracker"),
    "ChecklistGenerator": ("checklist_generator", "ChecklistGenerator"),
    "EntityCoordinator": ("entity_coordinator", "EntityCoordinator"),
    "YearComparator": ("year_comparator", "YearComparator"),
    "CalendarManager": ("calendar_manager", "CalendarManager"),
    "NotificationEngine": ("notification_engine", "NotificationEngine"),
    # Engine aliases (backward compat)
    "EntityCoordinatorEngine": ("entity_coordinator", "EntityCoordinatorEngine"),
    "YearComparatorEngine": ("year_comparator", "YearComparatorEngine"),
    "CalendarManagerEngine": ("calendar_manager", "CalendarManagerEngine"),
    # Service Facade
    "AnnualReviewSchedulerService": ("setup", "AnnualReviewSchedulerService"),
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
            f"greenlang.agents.eudr.annual_review_scheduler.{module_suffix}"
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
        "name": "Annual Review Scheduler Agent",
        "prd": "PRD-AGENT-EUDR-034",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["8", "10", "11", "12", "14", "29", "31"],
        "upstream_dependencies": [
            "AGENT-EUDR-033 (Continuous Monitoring)",
            "AGENT-EUDR-028 (Risk Assessment Engine)",
            "AGENT-EUDR-026 (Due Diligence Orchestrator)",
            "AGENT-EUDR-023 (Legal Compliance Verifier)",
            "AGENT-EUDR-030 (Documentation Generator)",
        ],
        "engines": [
            "ReviewCycleManager",
            "DeadlineTracker",
            "ChecklistGenerator",
            "EntityCoordinator",
            "YearComparator",
            "CalendarManager",
            "NotificationEngine",
        ],
        "engine_count": 7,
        "enum_count": 14,
        "core_model_count": 15,
        "sub_model_count": 9,
        "metrics_count": 40,
        "db_prefix": "gl_eudr_ars_",
        "metrics_prefix": "gl_eudr_ars_",
        "env_prefix": "GL_EUDR_ARS_",
    }
