# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Workflow Orchestration
====================================================

Pre-built workflow orchestrators for common CSRD compliance operations.
Each workflow coordinates GreenLang agents, data pipelines, and validation
engines into end-to-end compliance processes.

Workflows:
    - AnnualReportingWorkflow: Full annual CSRD reporting cycle
    - QuarterlyUpdateWorkflow: Lightweight quarterly data refresh
    - MaterialityAssessmentWorkflow: Standalone double materiality per ESRS 1
    - DataOnboardingWorkflow: Guided first-time data import
    - AuditPreparationWorkflow: Pre-audit compliance verification

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_001_csrd_starter.workflows.annual_reporting import (
    AnnualReportingWorkflow,
    AnnualReportingInput,
    AnnualReportingResult,
)
from packs.eu_compliance.PACK_001_csrd_starter.workflows.quarterly_update import (
    QuarterlyUpdateWorkflow,
    QuarterlyUpdateInput,
    QuarterlyUpdateResult,
)
from packs.eu_compliance.PACK_001_csrd_starter.workflows.materiality_assessment import (
    MaterialityAssessmentWorkflow,
    MaterialityAssessmentInput,
    MaterialityAssessmentResult,
)
from packs.eu_compliance.PACK_001_csrd_starter.workflows.data_onboarding import (
    DataOnboardingWorkflow,
    DataOnboardingInput,
    DataOnboardingResult,
)
from packs.eu_compliance.PACK_001_csrd_starter.workflows.audit_preparation import (
    AuditPreparationWorkflow,
    AuditPreparationInput,
    AuditPreparationResult,
)

__all__ = [
    "AnnualReportingWorkflow",
    "AnnualReportingInput",
    "AnnualReportingResult",
    "QuarterlyUpdateWorkflow",
    "QuarterlyUpdateInput",
    "QuarterlyUpdateResult",
    "MaterialityAssessmentWorkflow",
    "MaterialityAssessmentInput",
    "MaterialityAssessmentResult",
    "DataOnboardingWorkflow",
    "DataOnboardingInput",
    "DataOnboardingResult",
    "AuditPreparationWorkflow",
    "AuditPreparationInput",
    "AuditPreparationResult",
]
