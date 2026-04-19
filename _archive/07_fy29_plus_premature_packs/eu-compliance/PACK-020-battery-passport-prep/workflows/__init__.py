# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Workflows Module
=========================================================

This package provides 8 battery passport preparation workflows for the
PACK-020 Battery Passport Prep Pack. Each workflow implements a multi-phase
async pipeline for a specific EU Battery Regulation 2023/1542 requirement,
with SHA-256 provenance tracking and full audit trail support.

Workflows:
    1.  CarbonFootprintWorkflow         - Carbon footprint assessment (Art. 7)
    2.  RecycledContentWorkflow         - Recycled content tracking (Art. 8)
    3.  PassportCompilationWorkflow     - Digital passport compilation (Art. 77)
    4.  PerformanceTestingWorkflow      - Performance & durability testing (Art. 10-11)
    5.  DueDiligenceAssessmentWorkflow  - Supply chain due diligence (Art. 48)
    6.  LabellingVerificationWorkflow   - Labelling compliance (Art. 13-14)
    7.  EndOfLifePlanningWorkflow       - End-of-life management (Art. 59-62, 71)
    8.  RegulatorySubmissionWorkflow    - Regulatory submission (Art. 17-18)

Author: GreenLang Team
Version: 1.0.0
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-020"

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .carbon_footprint_assessment_workflow import CarbonFootprintWorkflow
    _loaded_workflows.append("CarbonFootprintWorkflow")
except ImportError as e:
    CarbonFootprintWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("CarbonFootprintWorkflow not available: %s", e)

try:
    from .recycled_content_tracking_workflow import RecycledContentWorkflow
    _loaded_workflows.append("RecycledContentWorkflow")
except ImportError as e:
    RecycledContentWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("RecycledContentWorkflow not available: %s", e)

try:
    from .passport_compilation_workflow import PassportCompilationWorkflow
    _loaded_workflows.append("PassportCompilationWorkflow")
except ImportError as e:
    PassportCompilationWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("PassportCompilationWorkflow not available: %s", e)

try:
    from .performance_testing_workflow import PerformanceTestingWorkflow
    _loaded_workflows.append("PerformanceTestingWorkflow")
except ImportError as e:
    PerformanceTestingWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("PerformanceTestingWorkflow not available: %s", e)

try:
    from .due_diligence_assessment_workflow import DueDiligenceAssessmentWorkflow
    _loaded_workflows.append("DueDiligenceAssessmentWorkflow")
except ImportError as e:
    DueDiligenceAssessmentWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("DueDiligenceAssessmentWorkflow not available: %s", e)

try:
    from .labelling_verification_workflow import LabellingVerificationWorkflow
    _loaded_workflows.append("LabellingVerificationWorkflow")
except ImportError as e:
    LabellingVerificationWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("LabellingVerificationWorkflow not available: %s", e)

try:
    from .end_of_life_planning_workflow import EndOfLifePlanningWorkflow
    _loaded_workflows.append("EndOfLifePlanningWorkflow")
except ImportError as e:
    EndOfLifePlanningWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("EndOfLifePlanningWorkflow not available: %s", e)

try:
    from .regulatory_submission_workflow import RegulatorySubmissionWorkflow
    _loaded_workflows.append("RegulatorySubmissionWorkflow")
except ImportError as e:
    RegulatorySubmissionWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("RegulatorySubmissionWorkflow not available: %s", e)


# ---------------------------------------------------------------------------
# Dynamic __all__
# ---------------------------------------------------------------------------

__all__: list[str] = [
    *_loaded_workflows,
    "get_loaded_workflows",
    "get_workflow_count",
    "get_regulation_workflow_mapping",
]


def get_loaded_workflows() -> list[str]:
    """Return list of successfully loaded workflow class names."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return count of loaded workflows."""
    return len(_loaded_workflows)


def get_regulation_workflow_mapping() -> Dict[str, str]:
    """Return mapping of EU Battery Regulation article to workflow class name."""
    return {
        "ART_7_CARBON_FOOTPRINT": "CarbonFootprintWorkflow",
        "ART_8_RECYCLED_CONTENT": "RecycledContentWorkflow",
        "ART_77_BATTERY_PASSPORT": "PassportCompilationWorkflow",
        "ART_10_11_PERFORMANCE": "PerformanceTestingWorkflow",
        "ART_48_DUE_DILIGENCE": "DueDiligenceAssessmentWorkflow",
        "ART_13_14_LABELLING": "LabellingVerificationWorkflow",
        "ART_59_62_71_END_OF_LIFE": "EndOfLifePlanningWorkflow",
        "ART_17_18_REGULATORY": "RegulatorySubmissionWorkflow",
    }
