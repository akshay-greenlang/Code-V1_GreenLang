# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Workflow Orchestration
================================================================

Quick-win energy efficiency workflow orchestrators for facility scanning,
measure prioritization, implementation planning, progress tracking,
reporting, and full end-to-end assessments. Each workflow coordinates
GreenLang calculation engines, data pipelines, and validation systems
into structured multi-phase processes with SHA-256 provenance hashing.

Workflows:
    - FacilityScanWorkflow: 4-phase facility scan with registration,
      quick-win scanning, initial savings estimation, and report generation.

    - PrioritizationWorkflow: 3-phase prioritization with financial
      analysis (NPV/IRR/payback), carbon assessment, and multi-criteria
      ranking using configurable weight profiles.

    - ImplementationPlanningWorkflow: 4-phase planning with sequencing,
      utility rebate matching, behavioral program design, and plan
      assembly with timeline and budget projections.

    - ProgressTrackingWorkflow: 3-phase tracking with data collection,
      IPMVP-based savings verification, and variance analysis with
      root cause identification.

    - ReportingWorkflow: 3-phase reporting with data aggregation,
      multi-format report generation, and distribution packaging.

    - FullAssessmentWorkflow: 6-phase end-to-end assessment orchestrating
      facility scan, financial analysis, carbon assessment, prioritization,
      implementation planning, and reporting into a single pipeline.

Author: GreenLang Team
Version: 33.0.0
"""

# ---------------------------------------------------------------------------
# Facility Scan Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.facility_scan_workflow import (
    FacilityScanWorkflow,
    FacilityScanInput,
    FacilityScanResult,
    QuickWinItem,
)

# ---------------------------------------------------------------------------
# Prioritization Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.prioritization_workflow import (
    PrioritizationWorkflow,
    PrioritizationInput,
    PrioritizationResult,
    RankedMeasure,
    ParetoPoint,
)

# ---------------------------------------------------------------------------
# Implementation Planning Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.implementation_planning_workflow import (
    ImplementationPlanningWorkflow,
    ImplementationPlanInput,
    ImplementationPlanResult,
    ImplementationPhase,
    RebateMatch,
)

# ---------------------------------------------------------------------------
# Progress Tracking Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.progress_tracking_workflow import (
    ProgressTrackingWorkflow,
    ProgressTrackingInput,
    ProgressTrackingResult,
    MeasureTrackingData,
    VarianceRecord,
)

# ---------------------------------------------------------------------------
# Reporting Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.reporting_workflow import (
    ReportingWorkflow,
    ReportingInput,
    ReportingResult,
    GeneratedReport,
)

# ---------------------------------------------------------------------------
# Full Assessment Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.workflows.full_assessment_workflow import (
    FullAssessmentWorkflow,
    FullAssessmentInput,
    FullAssessmentResult,
)

__all__ = [
    # --- Facility Scan Workflow ---
    "FacilityScanWorkflow",
    "FacilityScanInput",
    "FacilityScanResult",
    "QuickWinItem",
    # --- Prioritization Workflow ---
    "PrioritizationWorkflow",
    "PrioritizationInput",
    "PrioritizationResult",
    "RankedMeasure",
    "ParetoPoint",
    # --- Implementation Planning Workflow ---
    "ImplementationPlanningWorkflow",
    "ImplementationPlanInput",
    "ImplementationPlanResult",
    "ImplementationPhase",
    "RebateMatch",
    # --- Progress Tracking Workflow ---
    "ProgressTrackingWorkflow",
    "ProgressTrackingInput",
    "ProgressTrackingResult",
    "MeasureTrackingData",
    "VarianceRecord",
    # --- Reporting Workflow ---
    "ReportingWorkflow",
    "ReportingInput",
    "ReportingResult",
    "GeneratedReport",
    # --- Full Assessment Workflow ---
    "FullAssessmentWorkflow",
    "FullAssessmentInput",
    "FullAssessmentResult",
]
