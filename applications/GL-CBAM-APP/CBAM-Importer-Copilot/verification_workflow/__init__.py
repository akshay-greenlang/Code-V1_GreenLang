# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Verification Workflow Engine v1.1

Implements the CBAM verification lifecycle per Implementing Regulation (EU)
2023/1773 and the Omnibus Simplification Package (October 2025).

Starting 2026, CBAM declarations must be verified by an accredited verifier.
Site visits are annual in 2026 and biennial from 2027 onwards. The
materiality threshold for per-CN-code emission deviations is 5%.

Key regulatory references:
    - Regulation (EU) 2023/956 (CBAM Regulation), Articles 8 and 18
    - Implementing Regulation (EU) 2023/1773, Articles 10-18 (Verification)
    - EN ISO 14065:2020 (Requirements for GHG validation/verification bodies)
    - EN ISO 14064-3:2019 (GHG - Specification for verification)
    - Omnibus Simplification Package COM(2025) 508, Article 1(6)

Modules:
    verifier_registry: Accredited verifier management and COI checks
    verification_scheduler: Site visit scheduling and outcome recording
    materiality_assessor: 5% materiality threshold analysis

Example:
    >>> from verification_workflow import (
    ...     VerifierRegistryEngine,
    ...     VerificationSchedulerEngine,
    ...     MaterialityAssessorEngine,
    ... )
    >>> registry = VerifierRegistryEngine()
    >>> scheduler = VerificationSchedulerEngine(registry)
    >>> assessor = MaterialityAssessorEngine()

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

__version__ = "1.1.0"
__author__ = "GreenLang CBAM Team"

from verification_workflow.verifier_registry import (
    VerifierRegistryEngine,
    Verifier,
    AccreditationRecord,
    VerifierPerformance,
    ConflictOfInterestCheck,
)
from verification_workflow.verification_scheduler import (
    VerificationSchedulerEngine,
    SiteVisit,
    VisitType,
    VisitStatus,
    VisitOutcome,
    VisitFinding,
)
from verification_workflow.materiality_assessor import (
    MaterialityAssessorEngine,
    MaterialityResult,
    CnCodeMateriality,
    MaterialityTrend,
)

__all__ = [
    # Verifier registry
    "VerifierRegistryEngine",
    "Verifier",
    "AccreditationRecord",
    "VerifierPerformance",
    "ConflictOfInterestCheck",
    # Verification scheduler
    "VerificationSchedulerEngine",
    "SiteVisit",
    "VisitType",
    "VisitStatus",
    "VisitOutcome",
    "VisitFinding",
    # Materiality assessor
    "MaterialityAssessorEngine",
    "MaterialityResult",
    "CnCodeMateriality",
    "MaterialityTrend",
]

# CBAM verification constants
MATERIALITY_THRESHOLD_PCT = 5
ANNUAL_VISIT_REQUIRED_UNTIL = 2026  # Annual in 2026, biennial from 2027
