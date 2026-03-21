# -*- coding: utf-8 -*-
"""
PACK-023 SBTi Alignment Pack - Workflow Orchestration
========================================================

8 workflows implementing the complete SBTi alignment lifecycle from target
setting through full lifecycle management with submission readiness.

Workflows:
    1. TargetSettingWorkflow         -- 6 phases: inventory, screening, pathway,
                                       target definition, validation, summary
    2. ValidationWorkflow            -- 5 phases: evidence, near-term criteria,
                                       net-zero criteria, gap analysis, reporting
    3. Scope3AssessmentWorkflow      -- 5 phases: screening, materiality,
                                       coverage, targets, reporting
    4. SDAPathwayWorkflow            -- 6 phases: sector classification,
                                       benchmarks, convergence, cross-validation,
                                       ambition, reporting
    5. FLAGWorkflow                  -- 5 phases: trigger, commodity assessment,
                                       pathway, deforestation, reporting
    6. ProgressReviewWorkflow        -- 6 phases: data collection, tracking,
                                       variance analysis, recalculation,
                                       corrective actions, reporting
    7. FITargetWorkflow              -- 5 phases: portfolio inventory, asset class,
                                       coverage, engagement, validation
    8. FullSBTiLifecycleWorkflow     -- 10 phases: commitment through revalidation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-023"
__pack_name__: str = "SBTi Alignment Pack"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: TargetSettingWorkflow
# ---------------------------------------------------------------------------
_TARGET_SETTING_WF_SYMBOLS: list[str] = [
    "TargetSettingWorkflow",
    "TargetSettingConfig",
    "TargetSettingResult",
]
try:
    from .target_setting_workflow import (
        TargetSettingWorkflow,
        TargetSettingConfig,
        TargetSettingResult,
    )
    _loaded_workflows.append("TargetSettingWorkflow")
except ImportError as e:
    logger.debug("Workflow 1 (TargetSettingWorkflow) not available: %s", e)
    _TARGET_SETTING_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 2: ValidationWorkflow
# ---------------------------------------------------------------------------
_VALIDATION_WF_SYMBOLS: list[str] = [
    "ValidationWorkflow",
    "ValidationConfig",
    "ValidationResult",
]
try:
    from .validation_workflow import (
        ValidationWorkflow,
        ValidationConfig,
        ValidationResult,
    )
    _loaded_workflows.append("ValidationWorkflow")
except ImportError as e:
    logger.debug("Workflow 2 (ValidationWorkflow) not available: %s", e)
    _VALIDATION_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 3: Scope3AssessmentWorkflow
# ---------------------------------------------------------------------------
_SCOPE3_ASSESSMENT_WF_SYMBOLS: list[str] = [
    "Scope3AssessmentWorkflow",
    "Scope3AssessmentConfig",
    "Scope3AssessmentResult",
]
try:
    from .scope3_assessment_workflow import (
        Scope3AssessmentWorkflow,
        Scope3AssessmentConfig,
        Scope3AssessmentResult,
    )
    _loaded_workflows.append("Scope3AssessmentWorkflow")
except ImportError as e:
    logger.debug("Workflow 3 (Scope3AssessmentWorkflow) not available: %s", e)
    _SCOPE3_ASSESSMENT_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 4: SDAPathwayWorkflow
# ---------------------------------------------------------------------------
_SDA_PATHWAY_WF_SYMBOLS: list[str] = [
    "SDAPathwayWorkflow",
    "SDAPathwayConfig",
    "SDAPathwayResult",
]
try:
    from .sda_pathway_workflow import (
        SDAPathwayWorkflow,
        SDAPathwayConfig,
        SDAPathwayResult,
    )
    _loaded_workflows.append("SDAPathwayWorkflow")
except ImportError as e:
    logger.debug("Workflow 4 (SDAPathwayWorkflow) not available: %s", e)
    _SDA_PATHWAY_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 5: FLAGWorkflow
# ---------------------------------------------------------------------------
_FLAG_WF_SYMBOLS: list[str] = [
    "FLAGWorkflow",
    "FLAGWorkflowConfig",
    "FLAGWorkflowResult",
]
try:
    from .flag_workflow import (
        FLAGWorkflow,
        FLAGWorkflowConfig,
        FLAGWorkflowResult,
    )
    _loaded_workflows.append("FLAGWorkflow")
except ImportError as e:
    logger.debug("Workflow 5 (FLAGWorkflow) not available: %s", e)
    _FLAG_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 6: ProgressReviewWorkflow
# ---------------------------------------------------------------------------
_PROGRESS_REVIEW_WF_SYMBOLS: list[str] = [
    "ProgressReviewWorkflow",
    "ProgressReviewConfig",
    "ProgressReviewResult",
]
try:
    from .progress_review_workflow import (
        ProgressReviewWorkflow,
        ProgressReviewConfig,
        ProgressReviewResult,
    )
    _loaded_workflows.append("ProgressReviewWorkflow")
except ImportError as e:
    logger.debug("Workflow 6 (ProgressReviewWorkflow) not available: %s", e)
    _PROGRESS_REVIEW_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 7: FITargetWorkflow
# ---------------------------------------------------------------------------
_FI_TARGET_WF_SYMBOLS: list[str] = [
    "FITargetWorkflow",
    "FITargetWorkflowConfig",
    "FITargetWorkflowResult",
]
try:
    from .fi_target_workflow import (
        FITargetWorkflow,
        FITargetWorkflowConfig,
        FITargetWorkflowResult,
    )
    _loaded_workflows.append("FITargetWorkflow")
except ImportError as e:
    logger.debug("Workflow 7 (FITargetWorkflow) not available: %s", e)
    _FI_TARGET_WF_SYMBOLS = []

# ---------------------------------------------------------------------------
# Workflow 8: FullSBTiLifecycleWorkflow
# ---------------------------------------------------------------------------
_FULL_LIFECYCLE_WF_SYMBOLS: list[str] = [
    "FullSBTiLifecycleWorkflow",
    "LifecycleConfig",
    "LifecycleResult",
]
try:
    from .full_sbti_lifecycle_workflow import (
        FullSBTiLifecycleWorkflow,
        LifecycleConfig,
        LifecycleResult,
    )
    _loaded_workflows.append("FullSBTiLifecycleWorkflow")
except ImportError as e:
    logger.debug("Workflow 8 (FullSBTiLifecycleWorkflow) not available: %s", e)
    _FULL_LIFECYCLE_WF_SYMBOLS = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__workflows_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_TARGET_SETTING_WF_SYMBOLS,
    *_VALIDATION_WF_SYMBOLS,
    *_SCOPE3_ASSESSMENT_WF_SYMBOLS,
    *_SDA_PATHWAY_WF_SYMBOLS,
    *_FLAG_WF_SYMBOLS,
    *_PROGRESS_REVIEW_WF_SYMBOLS,
    *_FI_TARGET_WF_SYMBOLS,
    *_FULL_LIFECYCLE_WF_SYMBOLS,
]


def get_loaded_workflows() -> list[str]:
    """Return names of successfully loaded workflow classes."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return number of successfully loaded workflows."""
    return len(_loaded_workflows)


logger.info(
    "PACK-023 SBTi Alignment workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
