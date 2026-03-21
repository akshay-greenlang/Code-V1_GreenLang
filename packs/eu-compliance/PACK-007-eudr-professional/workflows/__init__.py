# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Workflow Layer
===================================================

10 compliance workflows for EUDR Professional Pack providing
advanced risk modeling, supply chain deep mapping, continuous
monitoring, supplier benchmarking, audit preparation, protected
area assessment, regulatory change response, multi-operator
onboarding, annual compliance review, and grievance resolution.

Workflows:
    1. AdvancedRiskModelingWorkflow       - Monte Carlo simulation and scenario analysis
    2. SupplyChainDeepMappingWorkflow     - Multi-tier supply chain network mapping
    3. ContinuousMonitoringWorkflow       - 24/7 real-time compliance monitoring
    4. SupplierBenchmarkingWorkflow       - Industry-relative supplier scoring
    5. AuditPreparationWorkflow           - CA inspection readiness and evidence assembly
    6. ProtectedAreaAssessmentWorkflow    - WDPA/KBA/indigenous territory analysis
    7. RegulatoryChangeResponseWorkflow   - EUR-Lex monitoring and impact assessment
    8. MultiOperatorOnboardingWorkflow    - Multi-entity operator onboarding
    9. AnnualComplianceReviewWorkflow     - Year-end compliance assessment
    10. GrievanceResolutionWorkflow       - Stakeholder complaint resolution

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-007 EUDR Professional Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-007"
__pack_name__ = "EUDR Professional Pack"

# ---------------------------------------------------------------------------
# 1. Advanced Risk Modeling Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.advanced_risk_modeling import (
    AdvancedRiskModelingWorkflow,
    AdvancedRiskModelingConfig,
    WorkflowResult as AdvancedRiskModelingResult,
    WorkflowContext as AdvancedRiskModelingContext,
)

# ---------------------------------------------------------------------------
# 2. Supply Chain Deep Mapping Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.supply_chain_deep_mapping import (
    SupplyChainDeepMappingWorkflow,
    SupplyChainDeepMappingConfig,
    WorkflowResult as SupplyChainDeepMappingResult,
    WorkflowContext as SupplyChainDeepMappingContext,
)

# ---------------------------------------------------------------------------
# 3. Continuous Monitoring Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.continuous_monitoring_workflow import (
    ContinuousMonitoringWorkflow,
    ContinuousMonitoringConfig,
    WorkflowResult as ContinuousMonitoringResult,
    WorkflowContext as ContinuousMonitoringContext,
)

# ---------------------------------------------------------------------------
# 4. Supplier Benchmarking Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.supplier_benchmarking_workflow import (
    SupplierBenchmarkingWorkflow,
    SupplierBenchmarkingConfig,
    WorkflowResult as SupplierBenchmarkingResult,
    WorkflowContext as SupplierBenchmarkingContext,
)

# ---------------------------------------------------------------------------
# 5. Audit Preparation Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.audit_preparation import (
    AuditPreparationWorkflow,
    AuditPreparationConfig,
    WorkflowResult as AuditPreparationResult,
    WorkflowContext as AuditPreparationContext,
)

# ---------------------------------------------------------------------------
# 6. Protected Area Assessment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.protected_area_assessment import (
    ProtectedAreaAssessmentWorkflow,
    ProtectedAreaAssessmentConfig,
    WorkflowResult as ProtectedAreaAssessmentResult,
    WorkflowContext as ProtectedAreaAssessmentContext,
)

# ---------------------------------------------------------------------------
# 7. Regulatory Change Response Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.regulatory_change_response import (
    RegulatoryChangeResponseWorkflow,
    RegulatoryChangeResponseConfig,
    WorkflowResult as RegulatoryChangeResponseResult,
    WorkflowContext as RegulatoryChangeResponseContext,
)

# ---------------------------------------------------------------------------
# 8. Multi-Operator Onboarding Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.multi_operator_onboarding import (
    MultiOperatorOnboardingWorkflow,
    MultiOperatorOnboardingConfig,
    WorkflowResult as MultiOperatorOnboardingResult,
    WorkflowContext as MultiOperatorOnboardingContext,
)

# ---------------------------------------------------------------------------
# 9. Annual Compliance Review Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.annual_compliance_review import (
    AnnualComplianceReviewWorkflow,
    AnnualComplianceReviewConfig,
    WorkflowResult as AnnualComplianceReviewResult,
    WorkflowContext as AnnualComplianceReviewContext,
)

# ---------------------------------------------------------------------------
# 10. Grievance Resolution Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_007_eudr_professional.workflows.grievance_resolution import (
    GrievanceResolutionWorkflow,
    GrievanceResolutionConfig,
    WorkflowResult as GrievanceResolutionResult,
    WorkflowContext as GrievanceResolutionContext,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- 1. Advanced Risk Modeling ---
    "AdvancedRiskModelingWorkflow",
    "AdvancedRiskModelingConfig",
    "AdvancedRiskModelingResult",
    "AdvancedRiskModelingContext",
    # --- 2. Supply Chain Deep Mapping ---
    "SupplyChainDeepMappingWorkflow",
    "SupplyChainDeepMappingConfig",
    "SupplyChainDeepMappingResult",
    "SupplyChainDeepMappingContext",
    # --- 3. Continuous Monitoring ---
    "ContinuousMonitoringWorkflow",
    "ContinuousMonitoringConfig",
    "ContinuousMonitoringResult",
    "ContinuousMonitoringContext",
    # --- 4. Supplier Benchmarking ---
    "SupplierBenchmarkingWorkflow",
    "SupplierBenchmarkingConfig",
    "SupplierBenchmarkingResult",
    "SupplierBenchmarkingContext",
    # --- 5. Audit Preparation ---
    "AuditPreparationWorkflow",
    "AuditPreparationConfig",
    "AuditPreparationResult",
    "AuditPreparationContext",
    # --- 6. Protected Area Assessment ---
    "ProtectedAreaAssessmentWorkflow",
    "ProtectedAreaAssessmentConfig",
    "ProtectedAreaAssessmentResult",
    "ProtectedAreaAssessmentContext",
    # --- 7. Regulatory Change Response ---
    "RegulatoryChangeResponseWorkflow",
    "RegulatoryChangeResponseConfig",
    "RegulatoryChangeResponseResult",
    "RegulatoryChangeResponseContext",
    # --- 8. Multi-Operator Onboarding ---
    "MultiOperatorOnboardingWorkflow",
    "MultiOperatorOnboardingConfig",
    "MultiOperatorOnboardingResult",
    "MultiOperatorOnboardingContext",
    # --- 9. Annual Compliance Review ---
    "AnnualComplianceReviewWorkflow",
    "AnnualComplianceReviewConfig",
    "AnnualComplianceReviewResult",
    "AnnualComplianceReviewContext",
    # --- 10. Grievance Resolution ---
    "GrievanceResolutionWorkflow",
    "GrievanceResolutionConfig",
    "GrievanceResolutionResult",
    "GrievanceResolutionContext",
]
