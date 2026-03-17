# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Workflow Orchestration
=====================================================

Pre-built workflow orchestrators for CBAM (Carbon Border Adjustment Mechanism)
compliance operations. Each workflow coordinates GreenLang agents, calculation
engines, and validation pipelines into end-to-end CBAM processes aligned with
EU Regulation 2023/956 and Implementing Regulation 2023/1773.

Workflows:
    - QuarterlyReportingWorkflow: 7-phase quarterly CBAM report cycle
    - AnnualDeclarationWorkflow: 8-phase annual certificate declaration
    - SupplierOnboardingWorkflow: 5-phase supplier registration and data quality
    - CertificateManagementWorkflow: 4-phase certificate lifecycle management
    - VerificationCycleWorkflow: 5-phase verification engagement
    - DeMinimisAssessmentWorkflow: 3-phase de minimis threshold assessment
    - DataCollectionWorkflow: 4-phase ongoing data intake orchestration

Regulatory Context:
    CBAM transitional period runs from October 2023 through December 2025,
    with quarterly reporting obligations. The definitive period begins
    January 2026 with certificate purchase/surrender obligations and
    annual declarations due by May 31 each year.

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_004_cbam_readiness.workflows.quarterly_reporting import (
    QuarterlyReportingWorkflow,
    QuarterlyReportResult,
    PhaseResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.annual_declaration import (
    AnnualDeclarationWorkflow,
    AnnualDeclarationResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.supplier_onboarding import (
    SupplierOnboardingWorkflow,
    SupplierOnboardingResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.certificate_management import (
    CertificateManagementWorkflow,
    CertificateManagementResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.verification_cycle import (
    VerificationCycleWorkflow,
    VerificationResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.deminimis_assessment import (
    DeMinimisAssessmentWorkflow,
    DeMinimisResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.workflows.data_collection import (
    DataCollectionWorkflow,
    DataCollectionResult,
)

__all__ = [
    # Quarterly Reporting
    "QuarterlyReportingWorkflow",
    "QuarterlyReportResult",
    "PhaseResult",
    # Annual Declaration
    "AnnualDeclarationWorkflow",
    "AnnualDeclarationResult",
    # Supplier Onboarding
    "SupplierOnboardingWorkflow",
    "SupplierOnboardingResult",
    # Certificate Management
    "CertificateManagementWorkflow",
    "CertificateManagementResult",
    # Verification Cycle
    "VerificationCycleWorkflow",
    "VerificationResult",
    # De Minimis Assessment
    "DeMinimisAssessmentWorkflow",
    "DeMinimisResult",
    # Data Collection
    "DataCollectionWorkflow",
    "DataCollectionResult",
]
