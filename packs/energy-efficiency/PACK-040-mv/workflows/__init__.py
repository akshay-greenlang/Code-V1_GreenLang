# -*- coding: utf-8 -*-
"""
PACK-040 Measurement & Verification (M&V) Pack - Workflows Module
====================================================================

Orchestration workflows for comprehensive M&V covering baseline development,
M&V plan creation, IPMVP option selection, post-installation verification,
savings verification, annual reporting, persistence tracking, and the full
M&V lifecycle.

Workflows:
    1. BaselineDevelopmentWorkflow   - 4-phase baseline regression and validation
    2. MVPlanWorkflow                - 4-phase M&V plan development
    3. OptionSelectionWorkflow       - 3-phase IPMVP option evaluation
    4. PostInstallationWorkflow      - 4-phase post-install verification
    5. SavingsVerificationWorkflow   - 4-phase savings calculation and uncertainty
    6. AnnualReportingWorkflow       - 3-phase annual report and compliance
    7. PersistenceTrackingWorkflow   - 3-phase degradation analysis and alerts
    8. FullMVWorkflow                - 8-phase complete M&V lifecycle

Regulatory references:
    - IPMVP Core Concepts (EVO 10000-1:2022)
    - ASHRAE Guideline 14-2014
    - ISO 50015:2014
    - FEMP M&V Guidelines 4.0
    - EU Energy Efficiency Directive (EED) Article 7

Pack Tier: Professional (PACK-040)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-040"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: Baseline Development
# ---------------------------------------------------------------------------
try:
    from .baseline_development_workflow import (
        BaselineDevelopmentWorkflow,
        BaselineDevelopmentInput,
        BaselineDevelopmentResult,
        DataRecord,
        ModelCandidate,
    )
    _loaded_workflows.append("BaselineDevelopmentWorkflow")
except ImportError as e:
    logger.debug("Workflow 1 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 2: M&V Plan
# ---------------------------------------------------------------------------
try:
    from .mv_plan_workflow import (
        MVPlanWorkflow,
        MVPlanInput,
        MVPlanResult,
        ECMDefinition,
    )
    _loaded_workflows.append("MVPlanWorkflow")
except ImportError as e:
    logger.debug("Workflow 2 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 3: Option Selection
# ---------------------------------------------------------------------------
try:
    from .option_selection_workflow import (
        OptionSelectionWorkflow,
        OptionSelectionInput,
        OptionSelectionResult,
        ECMCharacteristics,
        OptionScore,
    )
    _loaded_workflows.append("OptionSelectionWorkflow")
except ImportError as e:
    logger.debug("Workflow 3 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 4: Post-Installation
# ---------------------------------------------------------------------------
try:
    from .post_installation_workflow import (
        PostInstallationWorkflow,
        PostInstallationInput,
        PostInstallationResult,
        ECMInstallation,
        MeterInstallation,
    )
    _loaded_workflows.append("PostInstallationWorkflow")
except ImportError as e:
    logger.debug("Workflow 4 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 5: Savings Verification
# ---------------------------------------------------------------------------
try:
    from .savings_verification_workflow import (
        SavingsVerificationWorkflow,
        SavingsVerificationInput,
        SavingsVerificationResult,
        PeriodEnergyData,
        NonRoutineEvent,
    )
    _loaded_workflows.append("SavingsVerificationWorkflow")
except ImportError as e:
    logger.debug("Workflow 5 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 6: Annual Reporting
# ---------------------------------------------------------------------------
try:
    from .annual_reporting_workflow import (
        AnnualReportingWorkflow,
        AnnualReportingInput,
        AnnualReportingResult,
        AnnualSavingsRecord,
    )
    _loaded_workflows.append("AnnualReportingWorkflow")
except ImportError as e:
    logger.debug("Workflow 6 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 7: Persistence Tracking
# ---------------------------------------------------------------------------
try:
    from .persistence_tracking_workflow import (
        PersistenceTrackingWorkflow,
        PersistenceTrackingInput,
        PersistenceTrackingResult,
        PersistenceDataPoint,
    )
    _loaded_workflows.append("PersistenceTrackingWorkflow")
except ImportError as e:
    logger.debug("Workflow 7 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 8: Full M&V Lifecycle
# ---------------------------------------------------------------------------
try:
    from .full_mv_workflow import (
        FullMVWorkflow,
        FullMVInput,
        FullMVResult,
        ECMSummary,
    )
    _loaded_workflows.append("FullMVWorkflow")
except ImportError as e:
    logger.debug("Workflow 8 not available: %s", e)


__all__: list[str] = [
    "__version__",
    "__pack__",
    "__workflows_count__",
    # --- Baseline Development Workflow ---
    "BaselineDevelopmentWorkflow",
    "BaselineDevelopmentInput",
    "BaselineDevelopmentResult",
    "DataRecord",
    "ModelCandidate",
    # --- M&V Plan Workflow ---
    "MVPlanWorkflow",
    "MVPlanInput",
    "MVPlanResult",
    "ECMDefinition",
    # --- Option Selection Workflow ---
    "OptionSelectionWorkflow",
    "OptionSelectionInput",
    "OptionSelectionResult",
    "ECMCharacteristics",
    "OptionScore",
    # --- Post-Installation Workflow ---
    "PostInstallationWorkflow",
    "PostInstallationInput",
    "PostInstallationResult",
    "ECMInstallation",
    "MeterInstallation",
    # --- Savings Verification Workflow ---
    "SavingsVerificationWorkflow",
    "SavingsVerificationInput",
    "SavingsVerificationResult",
    "PeriodEnergyData",
    "NonRoutineEvent",
    # --- Annual Reporting Workflow ---
    "AnnualReportingWorkflow",
    "AnnualReportingInput",
    "AnnualReportingResult",
    "AnnualSavingsRecord",
    # --- Persistence Tracking Workflow ---
    "PersistenceTrackingWorkflow",
    "PersistenceTrackingInput",
    "PersistenceTrackingResult",
    "PersistenceDataPoint",
    # --- Full M&V Lifecycle Workflow ---
    "FullMVWorkflow",
    "FullMVInput",
    "FullMVResult",
    "ECMSummary",
]


def get_loaded_workflows() -> list[str]:
    """Return list of workflow class names that loaded successfully."""
    return list(_loaded_workflows)


logger.info(
    "PACK-040 M&V workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
