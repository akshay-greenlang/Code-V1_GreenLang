# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Workflows Module
======================================================

This package provides 8 EU Green Claims Directive preparation workflows
for the PACK-018 EU Green Claims Prep Pack. Each workflow implements a
multi-phase synchronous pipeline with SHA-256 provenance tracking and
full audit trail support.

Workflows:
    1. ClaimAssessmentWorkflow        - Assess environmental claims
    2. EvidenceCollectionWorkflow     - Collect and validate evidence
    3. LifecycleVerificationWorkflow  - Verify lifecycle-based claims (PEF)
    4. LabelAuditWorkflow             - Audit environmental labels
    5. GreenwashingScreeningWorkflow  - Screen for greenwashing patterns
    6. ComplianceGapWorkflow          - Identify compliance gaps
    7. RemediationPlanningWorkflow    - Plan remediation actions
    8. RegulatorySubmissionWorkflow   - Prepare regulatory submissions

Author: GreenLang Team
Version: 18.0.0
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

__version__: str = "18.0.0"
__pack__: str = "PACK-018"

_loaded_workflows: List[str] = []

# ---------------------------------------------------------------------------
# Workflow imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .claim_assessment_workflow import ClaimAssessmentWorkflow
    _loaded_workflows.append("ClaimAssessmentWorkflow")
except ImportError as e:
    ClaimAssessmentWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ClaimAssessmentWorkflow not available: %s", e)

try:
    from .evidence_collection_workflow import EvidenceCollectionWorkflow
    _loaded_workflows.append("EvidenceCollectionWorkflow")
except ImportError as e:
    EvidenceCollectionWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("EvidenceCollectionWorkflow not available: %s", e)

try:
    from .lifecycle_verification_workflow import LifecycleVerificationWorkflow
    _loaded_workflows.append("LifecycleVerificationWorkflow")
except ImportError as e:
    LifecycleVerificationWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("LifecycleVerificationWorkflow not available: %s", e)

try:
    from .label_audit_workflow import LabelAuditWorkflow
    _loaded_workflows.append("LabelAuditWorkflow")
except ImportError as e:
    LabelAuditWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("LabelAuditWorkflow not available: %s", e)

try:
    from .greenwashing_screening_workflow import GreenwashingScreeningWorkflow
    _loaded_workflows.append("GreenwashingScreeningWorkflow")
except ImportError as e:
    GreenwashingScreeningWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("GreenwashingScreeningWorkflow not available: %s", e)

try:
    from .compliance_gap_workflow import ComplianceGapWorkflow
    _loaded_workflows.append("ComplianceGapWorkflow")
except ImportError as e:
    ComplianceGapWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ComplianceGapWorkflow not available: %s", e)

try:
    from .remediation_planning_workflow import RemediationPlanningWorkflow
    _loaded_workflows.append("RemediationPlanningWorkflow")
except ImportError as e:
    RemediationPlanningWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("RemediationPlanningWorkflow not available: %s", e)

try:
    from .regulatory_submission_workflow import RegulatorySubmissionWorkflow
    _loaded_workflows.append("RegulatorySubmissionWorkflow")
except ImportError as e:
    RegulatorySubmissionWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("RegulatorySubmissionWorkflow not available: %s", e)


# ---------------------------------------------------------------------------
# Dynamic __all__
# ---------------------------------------------------------------------------

__all__: List[str] = [
    *_loaded_workflows,
    "get_loaded_workflows",
    "get_workflow_count",
    "get_workflow_mapping",
]


def get_loaded_workflows() -> List[str]:
    """Return list of successfully loaded workflow class names."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return count of loaded workflows."""
    return len(_loaded_workflows)


def get_workflow_mapping() -> Dict[str, str]:
    """Return mapping of workflow purpose to workflow class name."""
    return {
        "claim_assessment": "ClaimAssessmentWorkflow",
        "evidence_collection": "EvidenceCollectionWorkflow",
        "lifecycle_verification": "LifecycleVerificationWorkflow",
        "label_audit": "LabelAuditWorkflow",
        "greenwashing_screening": "GreenwashingScreeningWorkflow",
        "compliance_gap": "ComplianceGapWorkflow",
        "remediation_planning": "RemediationPlanningWorkflow",
        "regulatory_submission": "RegulatorySubmissionWorkflow",
    }
