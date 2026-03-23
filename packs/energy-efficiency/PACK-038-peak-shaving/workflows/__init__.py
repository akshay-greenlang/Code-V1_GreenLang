# -*- coding: utf-8 -*-
"""
PACK-038 Peak Shaving Pack - Workflows Module
====================================================

Orchestration workflows for peak shaving management covering load analysis,
peak assessment, BESS optimization, load shifting, coincident peak response,
implementation planning, M&V verification, and the full peak shaving lifecycle.

Workflows:
    1. LoadAnalysisWorkflow          - 4-phase load profile analysis
    2. PeakAssessmentWorkflow        - 4-phase peak attribution and strategy
    3. BESSOptimizationWorkflow      - 4-phase BESS sizing and financials
    4. LoadShiftWorkflow             - 3-phase load shifting and scheduling
    5. CPResponseWorkflow            - 3-phase coincident peak response
    6. ImplementationWorkflow        - 3-phase implementation planning
    7. VerificationWorkflow          - 3-phase M&V savings verification
    8. FullPeakShavingWorkflow       - 8-phase complete peak shaving lifecycle

Pack Tier: Professional (PACK-038)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-038"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: Load Analysis
# ---------------------------------------------------------------------------
try:
    from .load_analysis_workflow import (
        LoadAnalysisWorkflow,
        LoadAnalysisInput,
        LoadAnalysisResult,
        IntervalRecord,
    )
    _loaded_workflows.append("LoadAnalysisWorkflow")
except ImportError as e:
    logger.debug("Workflow 1 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 2: Peak Assessment
# ---------------------------------------------------------------------------
try:
    from .peak_assessment_workflow import (
        PeakAssessmentWorkflow,
        PeakAssessmentInput,
        PeakAssessmentResult,
        StrategyRecommendation,
    )
    _loaded_workflows.append("PeakAssessmentWorkflow")
except ImportError as e:
    logger.debug("Workflow 2 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 3: BESS Optimization
# ---------------------------------------------------------------------------
try:
    from .bess_optimization_workflow import (
        BESSOptimizationWorkflow,
        BESSOptimizationInput,
        BESSOptimizationResult,
    )
    _loaded_workflows.append("BESSOptimizationWorkflow")
except ImportError as e:
    logger.debug("Workflow 3 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 4: Load Shifting
# ---------------------------------------------------------------------------
try:
    from .load_shift_workflow import (
        LoadShiftWorkflow,
        LoadShiftInput,
        LoadShiftResult,
        ShiftableLoad,
    )
    _loaded_workflows.append("LoadShiftWorkflow")
except ImportError as e:
    logger.debug("Workflow 4 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 5: Coincident Peak Response
# ---------------------------------------------------------------------------
try:
    from .cp_response_workflow import (
        CPResponseWorkflow,
        CPResponseInput,
        CPResponseResult,
    )
    _loaded_workflows.append("CPResponseWorkflow")
except ImportError as e:
    logger.debug("Workflow 5 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 6: Implementation Planning
# ---------------------------------------------------------------------------
try:
    from .implementation_workflow import (
        ImplementationWorkflow,
        ImplementationInput,
        ImplementationResult,
    )
    _loaded_workflows.append("ImplementationWorkflow")
except ImportError as e:
    logger.debug("Workflow 6 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 7: M&V Verification
# ---------------------------------------------------------------------------
try:
    from .verification_workflow import (
        VerificationWorkflow,
        VerificationInput,
        VerificationResult,
    )
    _loaded_workflows.append("VerificationWorkflow")
except ImportError as e:
    logger.debug("Workflow 7 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 8: Full Peak Shaving Lifecycle
# ---------------------------------------------------------------------------
try:
    from .full_peak_shaving_workflow import (
        FullPeakShavingWorkflow,
        FullPeakShavingInput,
        FullPeakShavingResult,
    )
    _loaded_workflows.append("FullPeakShavingWorkflow")
except ImportError as e:
    logger.debug("Workflow 8 not available: %s", e)


__all__: list[str] = [
    "__version__",
    "__pack__",
    "__workflows_count__",
    # --- Load Analysis Workflow ---
    "LoadAnalysisWorkflow",
    "LoadAnalysisInput",
    "LoadAnalysisResult",
    "IntervalRecord",
    # --- Peak Assessment Workflow ---
    "PeakAssessmentWorkflow",
    "PeakAssessmentInput",
    "PeakAssessmentResult",
    "StrategyRecommendation",
    # --- BESS Optimization Workflow ---
    "BESSOptimizationWorkflow",
    "BESSOptimizationInput",
    "BESSOptimizationResult",
    # --- Load Shift Workflow ---
    "LoadShiftWorkflow",
    "LoadShiftInput",
    "LoadShiftResult",
    "ShiftableLoad",
    # --- CP Response Workflow ---
    "CPResponseWorkflow",
    "CPResponseInput",
    "CPResponseResult",
    # --- Implementation Workflow ---
    "ImplementationWorkflow",
    "ImplementationInput",
    "ImplementationResult",
    # --- Verification Workflow ---
    "VerificationWorkflow",
    "VerificationInput",
    "VerificationResult",
    # --- Full Peak Shaving Lifecycle Workflow ---
    "FullPeakShavingWorkflow",
    "FullPeakShavingInput",
    "FullPeakShavingResult",
]


def get_loaded_workflows() -> list[str]:
    """Return list of workflow class names that loaded successfully."""
    return list(_loaded_workflows)


logger.info(
    "PACK-038 Peak Shaving workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
