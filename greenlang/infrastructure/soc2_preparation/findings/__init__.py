# -*- coding: utf-8 -*-
"""
Findings Management - SEC-009 Phase 6

Comprehensive findings management for SOC 2 audit results. Provides finding
classification, tracking, remediation workflow, and closure verification
to ensure all audit findings are properly addressed.

Components:
    - FindingTracker: Create, classify, and track audit findings
    - RemediationWorkflow: Manage remediation lifecycle with SLAs
    - FindingClosure: Verify and close remediated findings

Finding Classifications:
    - EXCEPTION: Minor deviation from control procedure
    - CONTROL_DEFICIENCY: Control not operating effectively
    - SIGNIFICANT_DEFICIENCY: Material weakness in control
    - MATERIAL_WEAKNESS: Critical control failure

Example:
    >>> from greenlang.infrastructure.soc2_preparation.findings import (
    ...     FindingTracker,
    ...     RemediationWorkflow,
    ... )
    >>> tracker = FindingTracker()
    >>> finding = await tracker.create_finding(
    ...     FindingCreate(
    ...         title="MFA not enforced for admin accounts",
    ...         criterion_id="CC6.1",
    ...         description="5 admin accounts found without MFA enabled",
    ...     )
    ... )
    >>> classification = tracker.classify_finding(finding)
"""

from greenlang.infrastructure.soc2_preparation.findings.tracker import (
    FindingTracker,
    Finding,
    FindingCreate,
    FindingClassification,
    FindingStatus,
    FindingSummary,
    FindingAge,
)
from greenlang.infrastructure.soc2_preparation.findings.remediation import (
    RemediationWorkflow,
    RemediationPlan,
    RemediationState,
    RemediationProgress,
)
from greenlang.infrastructure.soc2_preparation.findings.closure import (
    FindingClosure,
    ClosureRequest,
    ClosureVerification,
)

__all__ = [
    # Tracker
    "FindingTracker",
    "Finding",
    "FindingCreate",
    "FindingClassification",
    "FindingStatus",
    "FindingSummary",
    "FindingAge",
    # Remediation
    "RemediationWorkflow",
    "RemediationPlan",
    "RemediationState",
    "RemediationProgress",
    # Closure
    "FindingClosure",
    "ClosureRequest",
    "ClosureVerification",
]
