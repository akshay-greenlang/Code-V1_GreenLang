# -*- coding: utf-8 -*-
"""
PACK-037 Demand Response Pack - Workflows Module
====================================================

Orchestration workflows for demand response management covering
flexibility assessment, program enrollment, event preparation,
event execution, settlement, DER optimization, reporting, and
the full DR lifecycle.

Workflows:
    1. FlexibilityAssessmentWorkflow   - 4-phase load flexibility assessment
    2. ProgramEnrollmentWorkflow       - 4-phase program selection and enrollment
    3. EventPreparationWorkflow        - 3-phase event preparation and dispatch
    4. EventExecutionWorkflow          - 4-phase real-time event management
    5. SettlementWorkflow              - 3-phase baseline and revenue settlement
    6. DEROptimizationWorkflow         - 3-phase DER coordination
    7. DRReportingWorkflow             - 3-phase report generation
    8. FullDRLifecycleWorkflow         - 8-phase complete DR lifecycle

Pack Tier: Professional (PACK-037)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-037"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: Flexibility Assessment
# ---------------------------------------------------------------------------
try:
    from .flexibility_assessment_workflow import (
        FlexibilityAssessmentWorkflow,
        FlexibilityAssessmentInput,
        FlexibilityAssessmentResult,
        LoadItem,
    )
    _loaded_workflows.append("FlexibilityAssessmentWorkflow")
except ImportError as e:
    logger.debug("Workflow 1 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 2: Program Enrollment
# ---------------------------------------------------------------------------
try:
    from .program_enrollment_workflow import (
        ProgramEnrollmentWorkflow,
        ProgramEnrollmentInput,
        ProgramEnrollmentResult,
        ProgramMatch,
    )
    _loaded_workflows.append("ProgramEnrollmentWorkflow")
except ImportError as e:
    logger.debug("Workflow 2 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 3: Event Preparation
# ---------------------------------------------------------------------------
try:
    from .event_preparation_workflow import (
        EventPreparationWorkflow,
        EventPreparationInput,
        EventPreparationResult,
        DispatchAction,
        PreConditionAction,
    )
    _loaded_workflows.append("EventPreparationWorkflow")
except ImportError as e:
    logger.debug("Workflow 3 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 4: Event Execution
# ---------------------------------------------------------------------------
try:
    from .event_execution_workflow import (
        EventExecutionWorkflow,
        EventExecutionInput,
        EventExecutionResult,
        CurtailmentAction,
        IntervalReading,
    )
    _loaded_workflows.append("EventExecutionWorkflow")
except ImportError as e:
    logger.debug("Workflow 4 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 5: Settlement
# ---------------------------------------------------------------------------
try:
    from .settlement_workflow import (
        SettlementWorkflow,
        SettlementInput,
        SettlementResult,
        BaselineResult,
        PerformanceMeasurement,
    )
    _loaded_workflows.append("SettlementWorkflow")
except ImportError as e:
    logger.debug("Workflow 5 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 6: DER Optimization
# ---------------------------------------------------------------------------
try:
    from .der_optimization_workflow import (
        DEROptimizationWorkflow,
        DEROptimizationInput,
        DEROptimizationResult,
        DERAsset,
        DERDispatchPlan,
    )
    _loaded_workflows.append("DEROptimizationWorkflow")
except ImportError as e:
    logger.debug("Workflow 6 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 7: DR Reporting
# ---------------------------------------------------------------------------
try:
    from .reporting_workflow import (
        DRReportingWorkflow,
        DRReportingInput,
        DRReportingResult,
        GeneratedReport,
    )
    _loaded_workflows.append("DRReportingWorkflow")
except ImportError as e:
    logger.debug("Workflow 7 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 8: Full DR Lifecycle
# ---------------------------------------------------------------------------
try:
    from .full_dr_lifecycle_workflow import (
        FullDRLifecycleWorkflow,
        FullDRLifecycleInput,
        FullDRLifecycleResult,
    )
    _loaded_workflows.append("FullDRLifecycleWorkflow")
except ImportError as e:
    logger.debug("Workflow 8 not available: %s", e)


__all__: list[str] = [
    "__version__",
    "__pack__",
    "__workflows_count__",
    # --- Flexibility Assessment Workflow ---
    "FlexibilityAssessmentWorkflow",
    "FlexibilityAssessmentInput",
    "FlexibilityAssessmentResult",
    "LoadItem",
    # --- Program Enrollment Workflow ---
    "ProgramEnrollmentWorkflow",
    "ProgramEnrollmentInput",
    "ProgramEnrollmentResult",
    "ProgramMatch",
    # --- Event Preparation Workflow ---
    "EventPreparationWorkflow",
    "EventPreparationInput",
    "EventPreparationResult",
    "DispatchAction",
    "PreConditionAction",
    # --- Event Execution Workflow ---
    "EventExecutionWorkflow",
    "EventExecutionInput",
    "EventExecutionResult",
    "CurtailmentAction",
    "IntervalReading",
    # --- Settlement Workflow ---
    "SettlementWorkflow",
    "SettlementInput",
    "SettlementResult",
    "BaselineResult",
    "PerformanceMeasurement",
    # --- DER Optimization Workflow ---
    "DEROptimizationWorkflow",
    "DEROptimizationInput",
    "DEROptimizationResult",
    "DERAsset",
    "DERDispatchPlan",
    # --- DR Reporting Workflow ---
    "DRReportingWorkflow",
    "DRReportingInput",
    "DRReportingResult",
    "GeneratedReport",
    # --- Full DR Lifecycle Workflow ---
    "FullDRLifecycleWorkflow",
    "FullDRLifecycleInput",
    "FullDRLifecycleResult",
]


def get_loaded_workflows() -> list[str]:
    """Return list of workflow class names that loaded successfully."""
    return list(_loaded_workflows)


logger.info(
    "PACK-037 Demand Response workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
