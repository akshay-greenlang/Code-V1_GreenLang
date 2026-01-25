"""
GL-013 PredictiveMaintenance - Maintenance Scheduling and Inventory Module

This module provides comprehensive maintenance scheduling, inventory planning,
work order generation, and closed-loop learning capabilities for predictive
maintenance operations.

Core Components:
- RiskScorer: Risk-based scoring using P(failure) * consequence
- MaintenanceScheduler: Production-aware maintenance scheduling
- InventoryPlanner: Parts demand forecasting and reorder proposals
- WorkOrderGenerator: Draft work orders with evidence references
- FeedbackProcessor: Closed-loop learning from CMMS outcomes

Example:
    >>> from maintenance import RiskScorer, MaintenanceScheduler
    >>> scorer = RiskScorer(config)
    >>> risk = scorer.calculate_risk(asset, rul_distribution)
    >>> scheduler = MaintenanceScheduler(config)
    >>> schedule = scheduler.optimize_schedule(risks, constraints)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .risk_scorer import (
        RiskScorer,
        RiskScore,
        RiskScorerConfig,
        AssetCriticality,
        ConsequenceFactors,
    )
    from .scheduler import (
        MaintenanceScheduler,
        MaintenanceSchedulerConfig,
        ScheduledMaintenance,
        ProductionWindow,
        TechnicianAvailability,
    )
    from .inventory_planner import (
        InventoryPlanner,
        InventoryPlannerConfig,
        PartsDemandForecast,
        ReorderProposal,
        BOMMapping,
    )
    from .work_order_generator import (
        WorkOrderGenerator,
        WorkOrderGeneratorConfig,
        DraftWorkOrder,
        ActionType,
        ApprovalWorkflow,
    )
    from .feedback_loop import (
        FeedbackProcessor,
        FeedbackProcessorConfig,
        CMMSClosureCode,
        TrainingLabel,
        AlertUsefulnessMetrics,
    )

__all__ = [
    # Risk Scorer
    "RiskScorer",
    "RiskScore",
    "RiskScorerConfig",
    "AssetCriticality",
    "ConsequenceFactors",
    # Scheduler
    "MaintenanceScheduler",
    "MaintenanceSchedulerConfig",
    "ScheduledMaintenance",
    "ProductionWindow",
    "TechnicianAvailability",
    # Inventory Planner
    "InventoryPlanner",
    "InventoryPlannerConfig",
    "PartsDemandForecast",
    "ReorderProposal",
    "BOMMapping",
    # Work Order Generator
    "WorkOrderGenerator",
    "WorkOrderGeneratorConfig",
    "DraftWorkOrder",
    "ActionType",
    "ApprovalWorkflow",
    # Feedback Loop
    "FeedbackProcessor",
    "FeedbackProcessorConfig",
    "CMMSClosureCode",
    "TrainingLabel",
    "AlertUsefulnessMetrics",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
