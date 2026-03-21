# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Workflow Layer
=======================================================

10 taxonomy compliance workflows for EU Taxonomy Alignment Pack
providing eligibility screening, alignment assessment, KPI
calculation, GAR computation, Article 8 disclosure generation,
gap analysis, CapEx planning, regulatory update response,
cross-framework alignment, and annual taxonomy review.

Workflows:
    1. EligibilityScreeningWorkflow       - Activity eligibility per 6 environmental objectives
    2. AlignmentAssessmentWorkflow        - Full SC + DNSH + MS alignment evaluation
    3. KPICalculationWorkflow             - Turnover/CapEx/OpEx ratio calculation
    4. GARCalculationWorkflow             - EBA Green Asset Ratio computation
    5. Article8DisclosureWorkflow         - Mandatory Article 8 disclosure generation
    6. GapAnalysisWorkflow                - Compliance gap identification and remediation
    7. CapExPlanWorkflow                  - CapEx plan generation for alignment improvement
    8. RegulatoryUpdateWorkflow           - Delegated Act change impact assessment
    9. CrossFrameworkAlignmentWorkflow    - CSRD/SFDR/EBA cross-framework alignment
    10. AnnualTaxonomyReviewWorkflow      - Year-end taxonomy assessment and trend analysis

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-008 EU Taxonomy Alignment Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-008"
__pack_name__ = "EU Taxonomy Alignment Pack"

# ---------------------------------------------------------------------------
# 1. Eligibility Screening Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.eligibility_screening import (
    EligibilityScreeningWorkflow,
    EligibilityScreeningConfig,
    WorkflowResult as EligibilityScreeningResult,
    WorkflowContext as EligibilityScreeningContext,
)

# ---------------------------------------------------------------------------
# 2. Alignment Assessment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.alignment_assessment import (
    AlignmentAssessmentWorkflow,
    AlignmentAssessmentConfig,
    WorkflowResult as AlignmentAssessmentResult,
    WorkflowContext as AlignmentAssessmentContext,
)

# ---------------------------------------------------------------------------
# 3. KPI Calculation Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.kpi_calculation import (
    KPICalculationWorkflow,
    KPICalculationConfig,
    WorkflowResult as KPICalculationResult,
    WorkflowContext as KPICalculationContext,
)

# ---------------------------------------------------------------------------
# 4. GAR Calculation Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.gar_calculation import (
    GARCalculationWorkflow,
    GARCalculationConfig,
    WorkflowResult as GARCalculationResult,
    WorkflowContext as GARCalculationContext,
)

# ---------------------------------------------------------------------------
# 5. Article 8 Disclosure Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.article8_disclosure import (
    Article8DisclosureWorkflow,
    Article8DisclosureConfig,
    WorkflowResult as Article8DisclosureResult,
    WorkflowContext as Article8DisclosureContext,
)

# ---------------------------------------------------------------------------
# 6. Gap Analysis Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.gap_analysis import (
    GapAnalysisWorkflow,
    GapAnalysisConfig,
    WorkflowResult as GapAnalysisResult,
    WorkflowContext as GapAnalysisContext,
)

# ---------------------------------------------------------------------------
# 7. CapEx Plan Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.capex_plan import (
    CapExPlanWorkflow,
    CapExPlanConfig,
    WorkflowResult as CapExPlanResult,
    WorkflowContext as CapExPlanContext,
)

# ---------------------------------------------------------------------------
# 8. Regulatory Update Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.regulatory_update import (
    RegulatoryUpdateWorkflow,
    RegulatoryUpdateConfig,
    WorkflowResult as RegulatoryUpdateResult,
    WorkflowContext as RegulatoryUpdateContext,
)

# ---------------------------------------------------------------------------
# 9. Cross-Framework Alignment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.cross_framework_alignment import (
    CrossFrameworkAlignmentWorkflow,
    CrossFrameworkConfig as CrossFrameworkAlignmentConfig,
    WorkflowResult as CrossFrameworkAlignmentResult,
    WorkflowContext as CrossFrameworkAlignmentContext,
)

# ---------------------------------------------------------------------------
# 10. Annual Taxonomy Review Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_008_eu_taxonomy_alignment.workflows.annual_taxonomy_review import (
    AnnualTaxonomyReviewWorkflow,
    AnnualTaxonomyReviewConfig,
    WorkflowResult as AnnualTaxonomyReviewResult,
    WorkflowContext as AnnualTaxonomyReviewContext,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- 1. Eligibility Screening ---
    "EligibilityScreeningWorkflow",
    "EligibilityScreeningConfig",
    "EligibilityScreeningResult",
    "EligibilityScreeningContext",
    # --- 2. Alignment Assessment ---
    "AlignmentAssessmentWorkflow",
    "AlignmentAssessmentConfig",
    "AlignmentAssessmentResult",
    "AlignmentAssessmentContext",
    # --- 3. KPI Calculation ---
    "KPICalculationWorkflow",
    "KPICalculationConfig",
    "KPICalculationResult",
    "KPICalculationContext",
    # --- 4. GAR Calculation ---
    "GARCalculationWorkflow",
    "GARCalculationConfig",
    "GARCalculationResult",
    "GARCalculationContext",
    # --- 5. Article 8 Disclosure ---
    "Article8DisclosureWorkflow",
    "Article8DisclosureConfig",
    "Article8DisclosureResult",
    "Article8DisclosureContext",
    # --- 6. Gap Analysis ---
    "GapAnalysisWorkflow",
    "GapAnalysisConfig",
    "GapAnalysisResult",
    "GapAnalysisContext",
    # --- 7. CapEx Plan ---
    "CapExPlanWorkflow",
    "CapExPlanConfig",
    "CapExPlanResult",
    "CapExPlanContext",
    # --- 8. Regulatory Update ---
    "RegulatoryUpdateWorkflow",
    "RegulatoryUpdateConfig",
    "RegulatoryUpdateResult",
    "RegulatoryUpdateContext",
    # --- 9. Cross-Framework Alignment ---
    "CrossFrameworkAlignmentWorkflow",
    "CrossFrameworkAlignmentConfig",
    "CrossFrameworkAlignmentResult",
    "CrossFrameworkAlignmentContext",
    # --- 10. Annual Taxonomy Review ---
    "AnnualTaxonomyReviewWorkflow",
    "AnnualTaxonomyReviewConfig",
    "AnnualTaxonomyReviewResult",
    "AnnualTaxonomyReviewContext",
]
