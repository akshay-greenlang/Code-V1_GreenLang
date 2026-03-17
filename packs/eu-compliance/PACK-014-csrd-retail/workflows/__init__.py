# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Workflow Orchestration
=====================================================================

Retail-specific workflow orchestrators for CSRD compliance operations.
Each workflow coordinates GreenLang agents, data pipelines, AI engines,
and validation systems into end-to-end retail compliance processes
covering store emissions, supply chains, packaging, product
sustainability, food waste, circular economy, ESRS disclosure,
and regulatory compliance.

Workflows:
    - StoreEmissionsWorkflow: 5-phase store-level emissions calculation
      with multi-store consolidation, refrigerant leak detection, fleet
      analysis, and Scope 1+2 aggregation with intensity KPIs.

    - SupplyChainAssessmentWorkflow: 5-phase supply chain ESG assessment
      with multi-tier supplier mapping, Scope 3 category estimation,
      hotspot identification, and supplier engagement planning.

    - PackagingComplianceWorkflow: 4-phase PPWR packaging assessment
      with recycled content gap analysis, EPR eco-modulation grading,
      labeling compliance checks, and material optimization roadmap.

    - ProductSustainabilityWorkflow: 4-phase product-level sustainability
      with PEF life-cycle assessment, ESPR Digital Product Passport
      readiness, ECGT green claims audit, and product scoring.

    - FoodWasteTrackingWorkflow: 4-phase food waste management with
      baseline establishment, category-level breakdown, reduction
      targeting vs EU 30% goal (2030), and progress reporting.

    - CircularEconomyWorkflow: 4-phase circular economy assessment with
      take-back program metrics, material recovery tracking, EPR scheme
      compliance, and circularity index calculation (MCI).

    - ESRSRetailDisclosureWorkflow: 4-phase ESRS disclosure generation
      with retail materiality assessment, datapoint collection, chapter
      generation (E1/E5/S2/S4), and audit evidence packaging.

    - RegulatoryComplianceWorkflow: 5-phase multi-regulation compliance
      with regulation mapping by sub-sector, gap assessment, action
      planning with priority scoring, and compliance dashboard.

Author: GreenLang Team
Version: 14.0.0
"""

# ---------------------------------------------------------------------------
# Store Emissions Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.store_emissions_workflow import (
    StoreEmissionsWorkflow,
    StoreEmissionsInput,
    StoreEmissionsResult,
    StoreEmissionsConfig,
    StoreData,
    StoreEmissionBreakdown,
    ConsolidatedKPIs,
    EnergyRecord,
    RefrigerantRecord,
    FleetRecord,
    ElectricityRecord,
)

# ---------------------------------------------------------------------------
# Supply Chain Assessment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.supply_chain_assessment_workflow import (
    SupplyChainAssessmentWorkflow,
    SupplyChainInput,
    SupplyChainResult,
    SupplierRecord,
    PurchasedGoodRecord,
    TransportRecord,
    CategoryEmission,
    Hotspot,
    EngagementAction,
)

# ---------------------------------------------------------------------------
# Packaging Compliance Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.packaging_compliance_workflow import (
    PackagingComplianceWorkflow,
    PackagingInput,
    PackagingResult,
    PackagingItem,
    RecycledContentTarget,
    RecycledContentGap,
    EPRFeeItem,
    LabelingResult,
)

# ---------------------------------------------------------------------------
# Product Sustainability Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.product_sustainability_workflow import (
    ProductSustainabilityWorkflow,
    ProductSustainabilityInput,
    ProductSustainabilityResult,
    ProductRecord,
    GreenClaim,
    DPPRecord,
    ClaimAuditResult,
    PEFResult,
)

# ---------------------------------------------------------------------------
# Food Waste Tracking Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.food_waste_tracking_workflow import (
    FoodWasteTrackingWorkflow,
    FoodWasteInput,
    FoodWasteResult,
    WasteRecord,
    BaselineData,
    CategoryBreakdown,
    ReductionTarget,
    WasteRecommendation,
)

# ---------------------------------------------------------------------------
# Circular Economy Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.circular_economy_workflow import (
    CircularEconomyWorkflow,
    CircularEconomyInput,
    CircularEconomyResult,
    TakeBackProgram,
    MaterialFlow,
    EPRData,
    TakeBackMetrics,
)

# ---------------------------------------------------------------------------
# ESRS Retail Disclosure Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.esrs_retail_disclosure_workflow import (
    ESRSRetailDisclosureWorkflow,
    ESRSDisclosureInput,
    ESRSDisclosureResult,
    MaterialityResult,
    ESRSDataPoint,
    DisclosureChapter,
    EvidencePackage,
)

# ---------------------------------------------------------------------------
# Regulatory Compliance Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.workflows.regulatory_compliance_workflow import (
    RegulatoryComplianceWorkflow,
    RegulatoryComplianceInput,
    RegulatoryComplianceResult,
    CompanyData,
    RegulationRecord,
    ComplianceAssessmentItem,
    ActionItem,
)

__all__ = [
    # --- Store Emissions Workflow ---
    "StoreEmissionsWorkflow",
    "StoreEmissionsInput",
    "StoreEmissionsResult",
    "StoreEmissionsConfig",
    "StoreData",
    "StoreEmissionBreakdown",
    "ConsolidatedKPIs",
    "EnergyRecord",
    "RefrigerantRecord",
    "FleetRecord",
    "ElectricityRecord",
    # --- Supply Chain Assessment Workflow ---
    "SupplyChainAssessmentWorkflow",
    "SupplyChainInput",
    "SupplyChainResult",
    "SupplierRecord",
    "PurchasedGoodRecord",
    "TransportRecord",
    "CategoryEmission",
    "Hotspot",
    "EngagementAction",
    # --- Packaging Compliance Workflow ---
    "PackagingComplianceWorkflow",
    "PackagingInput",
    "PackagingResult",
    "PackagingItem",
    "RecycledContentTarget",
    "RecycledContentGap",
    "EPRFeeItem",
    "LabelingResult",
    # --- Product Sustainability Workflow ---
    "ProductSustainabilityWorkflow",
    "ProductSustainabilityInput",
    "ProductSustainabilityResult",
    "ProductRecord",
    "GreenClaim",
    "DPPRecord",
    "ClaimAuditResult",
    "PEFResult",
    # --- Food Waste Tracking Workflow ---
    "FoodWasteTrackingWorkflow",
    "FoodWasteInput",
    "FoodWasteResult",
    "WasteRecord",
    "BaselineData",
    "CategoryBreakdown",
    "ReductionTarget",
    "WasteRecommendation",
    # --- Circular Economy Workflow ---
    "CircularEconomyWorkflow",
    "CircularEconomyInput",
    "CircularEconomyResult",
    "TakeBackProgram",
    "MaterialFlow",
    "EPRData",
    "TakeBackMetrics",
    # --- ESRS Retail Disclosure Workflow ---
    "ESRSRetailDisclosureWorkflow",
    "ESRSDisclosureInput",
    "ESRSDisclosureResult",
    "MaterialityResult",
    "ESRSDataPoint",
    "DisclosureChapter",
    "EvidencePackage",
    # --- Regulatory Compliance Workflow ---
    "RegulatoryComplianceWorkflow",
    "RegulatoryComplianceInput",
    "RegulatoryComplianceResult",
    "CompanyData",
    "RegulationRecord",
    "ComplianceAssessmentItem",
    "ActionItem",
]
