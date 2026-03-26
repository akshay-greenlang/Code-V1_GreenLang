# -*- coding: utf-8 -*-
"""
PACK-041 Scope 1-2 Complete Pack - Workflow Orchestration
=============================================================

Complete Scope 1 and Scope 2 GHG inventory workflow orchestrators for the
GHG Protocol Corporate Standard, ISO 14064-1:2018, and multi-framework
disclosure compliance. Each workflow coordinates GreenLang MRV agents,
DATA agents, calculation engines, and validation systems into end-to-end
inventory processes covering boundary definition, data collection, Scope 1
calculation (8 categories), Scope 2 dual-method calculation, inventory
consolidation with uncertainty, verification preparation, multi-framework
disclosure generation, and full end-to-end inventory orchestration.

Workflows:
    - BoundaryDefinitionWorkflow: 4-phase boundary definition with entity
      mapping, consolidation approach selection, source identification,
      and materiality assessment using sector benchmarks.

    - DataCollectionWorkflow: 4-phase data collection with requirement
      generation, multi-source ingestion via DATA agents, deterministic
      quality scoring (0-100), and gap resolution with remediation.

    - Scope1CalculationWorkflow: 4-phase Scope 1 calculation with MRV
      agent routing (001-008), parallel execution, boundary-adjusted
      consolidation, and cross-source double-counting reconciliation.

    - Scope2CalculationWorkflow: 4-phase Scope 2 dual-method calculation
      with contractual instrument validation, location+market parallel
      execution (MRV 009-012), GHG Protocol hierarchy allocation, and
      variance analysis.

    - InventoryConsolidationWorkflow: 4-phase consolidation with Scope 1
      aggregation, Scope 2 dual-method aggregation, IPCC uncertainty
      propagation (analytical + Monte Carlo), and total inventory
      generation with per-facility and per-entity breakdowns.

    - VerificationPreparationWorkflow: 4-phase verification preparation
      with audit trail compilation, SHA-256 provenance chain verification,
      ISO 14064-1 completeness check, and verification package generation.

    - DisclosureGenerationWorkflow: 4-phase multi-framework disclosure
      with requirement mapping (ESRS, CDP, TCFD, SBTi, GHG Protocol,
      ISO 14064, SEC, SB 253), template population, compliance validation,
      and framework-native output generation (XBRL, XML, PDF, Excel).

    - FullInventoryWorkflow: 8-phase end-to-end orchestrator invoking all
      sub-workflows in sequence with full data handoff, plus trend analysis
      comparing to base year and previous years.

Author: GreenLang Team
Version: 41.0.0
"""

from typing import Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Boundary Definition Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.boundary_definition_workflow import (
        BoundaryDefinitionWorkflow,
        BoundaryDefinitionInput,
        BoundaryDefinitionResult,
        EntityRecord,
        FacilityRecord,
        EntityBoundaryResult,
        SourceCategoryAssignment,
        MaterialityRecord,
        CompletenessReport,
        ConsolidationApproach,
        EntityType,
        MaterialityLevel,
    )
except ImportError:
    BoundaryDefinitionWorkflow = None  # type: ignore[assignment,misc]
    BoundaryDefinitionInput = None  # type: ignore[assignment,misc]
    BoundaryDefinitionResult = None  # type: ignore[assignment,misc]
    EntityRecord = None  # type: ignore[assignment,misc]
    FacilityRecord = None  # type: ignore[assignment,misc]
    EntityBoundaryResult = None  # type: ignore[assignment,misc]
    SourceCategoryAssignment = None  # type: ignore[assignment,misc]
    MaterialityRecord = None  # type: ignore[assignment,misc]
    CompletenessReport = None  # type: ignore[assignment,misc]
    ConsolidationApproach = None  # type: ignore[assignment,misc]
    EntityType = None  # type: ignore[assignment,misc]
    MaterialityLevel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Data Collection Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.data_collection_workflow import (
        DataCollectionWorkflow,
        DataCollectionInput,
        DataCollectionResult,
        DataRequirement,
        IngestedDataRecord,
        QualityScore,
        DataGap,
        FacilityDataSource,
        DataSourceType,
        DataQualityRating,
        GapSeverity,
        RemediationAction,
    )
except ImportError:
    DataCollectionWorkflow = None  # type: ignore[assignment,misc]
    DataCollectionInput = None  # type: ignore[assignment,misc]
    DataCollectionResult = None  # type: ignore[assignment,misc]
    DataRequirement = None  # type: ignore[assignment,misc]
    IngestedDataRecord = None  # type: ignore[assignment,misc]
    QualityScore = None  # type: ignore[assignment,misc]
    DataGap = None  # type: ignore[assignment,misc]
    FacilityDataSource = None  # type: ignore[assignment,misc]
    DataSourceType = None  # type: ignore[assignment,misc]
    DataQualityRating = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    RemediationAction = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Scope 1 Calculation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.scope1_calculation_workflow import (
        Scope1CalculationWorkflow,
        Scope1CalculationInput,
        Scope1CalculationResult,
        FacilityActivityData,
        BoundaryDef,
        AgentRoutingEntry,
        AgentExecutionResult,
        CategoryTotal,
        FacilityTotal,
        DoubleCountFlag,
        Scope1Category,
        GHGGas,
        DoubleCountType,
    )
except ImportError:
    Scope1CalculationWorkflow = None  # type: ignore[assignment,misc]
    Scope1CalculationInput = None  # type: ignore[assignment,misc]
    Scope1CalculationResult = None  # type: ignore[assignment,misc]
    FacilityActivityData = None  # type: ignore[assignment,misc]
    BoundaryDef = None  # type: ignore[assignment,misc]
    AgentRoutingEntry = None  # type: ignore[assignment,misc]
    AgentExecutionResult = None  # type: ignore[assignment,misc]
    CategoryTotal = None  # type: ignore[assignment,misc]
    FacilityTotal = None  # type: ignore[assignment,misc]
    DoubleCountFlag = None  # type: ignore[assignment,misc]
    Scope1Category = None  # type: ignore[assignment,misc]
    GHGGas = None  # type: ignore[assignment,misc]
    DoubleCountType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Scope 2 Calculation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.scope2_calculation_workflow import (
        Scope2CalculationWorkflow,
        Scope2CalculationInput,
        Scope2CalculationResult,
        ContractualInstrument,
        FacilityConsumption,
        FacilityDualResult,
        InstrumentAllocation,
        VarianceAnalysis,
        InstrumentType,
        InstrumentQuality,
        Scope2Method,
        EnergyType,
    )
except ImportError:
    Scope2CalculationWorkflow = None  # type: ignore[assignment,misc]
    Scope2CalculationInput = None  # type: ignore[assignment,misc]
    Scope2CalculationResult = None  # type: ignore[assignment,misc]
    ContractualInstrument = None  # type: ignore[assignment,misc]
    FacilityConsumption = None  # type: ignore[assignment,misc]
    FacilityDualResult = None  # type: ignore[assignment,misc]
    InstrumentAllocation = None  # type: ignore[assignment,misc]
    VarianceAnalysis = None  # type: ignore[assignment,misc]
    InstrumentType = None  # type: ignore[assignment,misc]
    InstrumentQuality = None  # type: ignore[assignment,misc]
    Scope2Method = None  # type: ignore[assignment,misc]
    EnergyType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Inventory Consolidation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.inventory_consolidation_workflow import (
        InventoryConsolidationWorkflow,
        InventoryConsolidationInput,
        InventoryConsolidationResult,
        Scope1InputData,
        Scope2InputData,
        UncertaintyConfig,
        Scope1Summary,
        Scope2Summary,
        FacilityInventory,
        EntityInventory,
        UncertaintyBounds,
        GasBreakdown,
        CategoryBreakdown,
        UncertaintyMethod,
        ConfidenceLevel,
    )
except ImportError:
    InventoryConsolidationWorkflow = None  # type: ignore[assignment,misc]
    InventoryConsolidationInput = None  # type: ignore[assignment,misc]
    InventoryConsolidationResult = None  # type: ignore[assignment,misc]
    Scope1InputData = None  # type: ignore[assignment,misc]
    Scope2InputData = None  # type: ignore[assignment,misc]
    UncertaintyConfig = None  # type: ignore[assignment,misc]
    Scope1Summary = None  # type: ignore[assignment,misc]
    Scope2Summary = None  # type: ignore[assignment,misc]
    FacilityInventory = None  # type: ignore[assignment,misc]
    EntityInventory = None  # type: ignore[assignment,misc]
    UncertaintyBounds = None  # type: ignore[assignment,misc]
    GasBreakdown = None  # type: ignore[assignment,misc]
    CategoryBreakdown = None  # type: ignore[assignment,misc]
    UncertaintyMethod = None  # type: ignore[assignment,misc]
    ConfidenceLevel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Verification Preparation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.verification_preparation_workflow import (
        VerificationPreparationWorkflow,
        VerificationInput,
        VerificationResult,
        AuditTrailEntry,
        ProvenanceHashRecord,
        CompletenessRequirement,
        VerificationPackageSection,
        MethodologyDescription,
        VerificationLevel,
        CompletenessStatus,
        HashVerificationStatus,
    )
except ImportError:
    VerificationPreparationWorkflow = None  # type: ignore[assignment,misc]
    VerificationInput = None  # type: ignore[assignment,misc]
    VerificationResult = None  # type: ignore[assignment,misc]
    AuditTrailEntry = None  # type: ignore[assignment,misc]
    ProvenanceHashRecord = None  # type: ignore[assignment,misc]
    CompletenessRequirement = None  # type: ignore[assignment,misc]
    VerificationPackageSection = None  # type: ignore[assignment,misc]
    MethodologyDescription = None  # type: ignore[assignment,misc]
    VerificationLevel = None  # type: ignore[assignment,misc]
    CompletenessStatus = None  # type: ignore[assignment,misc]
    HashVerificationStatus = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Disclosure Generation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.disclosure_generation_workflow import (
        DisclosureGenerationWorkflow,
        DisclosureInput,
        DisclosureResult,
        FrameworkRequirement,
        FrameworkTemplate,
        ComplianceScore,
        GapAnalysis,
        FrameworkOutput,
        DisclosureFramework,
        OutputFormat,
        ComplianceLevel,
    )
except ImportError:
    DisclosureGenerationWorkflow = None  # type: ignore[assignment,misc]
    DisclosureInput = None  # type: ignore[assignment,misc]
    DisclosureResult = None  # type: ignore[assignment,misc]
    FrameworkRequirement = None  # type: ignore[assignment,misc]
    FrameworkTemplate = None  # type: ignore[assignment,misc]
    ComplianceScore = None  # type: ignore[assignment,misc]
    GapAnalysis = None  # type: ignore[assignment,misc]
    FrameworkOutput = None  # type: ignore[assignment,misc]
    DisclosureFramework = None  # type: ignore[assignment,misc]
    OutputFormat = None  # type: ignore[assignment,misc]
    ComplianceLevel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Inventory Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_041_scope_1_2_complete.workflows.full_inventory_workflow import (
        FullInventoryWorkflow,
        FullInventoryInput,
        FullInventoryResult,
        OrganizationStructure,
        BaseYearConfig,
        TrendResult,
        TrendDirection,
    )
except ImportError:
    FullInventoryWorkflow = None  # type: ignore[assignment,misc]
    FullInventoryInput = None  # type: ignore[assignment,misc]
    FullInventoryResult = None  # type: ignore[assignment,misc]
    OrganizationStructure = None  # type: ignore[assignment,misc]
    BaseYearConfig = None  # type: ignore[assignment,misc]
    TrendResult = None  # type: ignore[assignment,misc]
    TrendDirection = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Boundary Definition types
BoundaryInput = BoundaryDefinitionInput
BoundaryOutput = BoundaryDefinitionResult

# Data Collection types
DataInput = DataCollectionInput
DataOutput = DataCollectionResult

# Scope 1 Calculation types
S1Input = Scope1CalculationInput
S1Output = Scope1CalculationResult

# Scope 2 Calculation types
S2Input = Scope2CalculationInput
S2Output = Scope2CalculationResult

# Inventory Consolidation types
ConsolidationInput = InventoryConsolidationInput
ConsolidationOutput = InventoryConsolidationResult

# Verification types
VerifInput = VerificationInput
VerifOutput = VerificationResult

# Disclosure types
DiscInput = DisclosureInput
DiscOutput = DisclosureResult

# Full Inventory types
FullInput = FullInventoryInput
FullOutput = FullInventoryResult


# ---------------------------------------------------------------------------
# Loaded Workflow Helper
# ---------------------------------------------------------------------------

def get_loaded_workflows() -> Dict[str, Optional[type]]:
    """
    Return a dictionary of workflow names to their classes.

    Workflows that failed to import will have None values.

    Returns:
        Dict mapping workflow name to workflow class or None.
    """
    return {
        "boundary_definition": BoundaryDefinitionWorkflow,
        "data_collection": DataCollectionWorkflow,
        "scope1_calculation": Scope1CalculationWorkflow,
        "scope2_calculation": Scope2CalculationWorkflow,
        "inventory_consolidation": InventoryConsolidationWorkflow,
        "verification_preparation": VerificationPreparationWorkflow,
        "disclosure_generation": DisclosureGenerationWorkflow,
        "full_inventory": FullInventoryWorkflow,
    }


__all__ = [
    # --- Boundary Definition Workflow ---
    "BoundaryDefinitionWorkflow",
    "BoundaryDefinitionInput",
    "BoundaryDefinitionResult",
    "EntityRecord",
    "FacilityRecord",
    "EntityBoundaryResult",
    "SourceCategoryAssignment",
    "MaterialityRecord",
    "CompletenessReport",
    "ConsolidationApproach",
    "EntityType",
    "MaterialityLevel",
    # --- Data Collection Workflow ---
    "DataCollectionWorkflow",
    "DataCollectionInput",
    "DataCollectionResult",
    "DataRequirement",
    "IngestedDataRecord",
    "QualityScore",
    "DataGap",
    "FacilityDataSource",
    "DataSourceType",
    "DataQualityRating",
    "GapSeverity",
    "RemediationAction",
    # --- Scope 1 Calculation Workflow ---
    "Scope1CalculationWorkflow",
    "Scope1CalculationInput",
    "Scope1CalculationResult",
    "FacilityActivityData",
    "BoundaryDef",
    "AgentRoutingEntry",
    "AgentExecutionResult",
    "CategoryTotal",
    "FacilityTotal",
    "DoubleCountFlag",
    "Scope1Category",
    "GHGGas",
    "DoubleCountType",
    # --- Scope 2 Calculation Workflow ---
    "Scope2CalculationWorkflow",
    "Scope2CalculationInput",
    "Scope2CalculationResult",
    "ContractualInstrument",
    "FacilityConsumption",
    "FacilityDualResult",
    "InstrumentAllocation",
    "VarianceAnalysis",
    "InstrumentType",
    "InstrumentQuality",
    "Scope2Method",
    "EnergyType",
    # --- Inventory Consolidation Workflow ---
    "InventoryConsolidationWorkflow",
    "InventoryConsolidationInput",
    "InventoryConsolidationResult",
    "Scope1InputData",
    "Scope2InputData",
    "UncertaintyConfig",
    "Scope1Summary",
    "Scope2Summary",
    "FacilityInventory",
    "EntityInventory",
    "UncertaintyBounds",
    "GasBreakdown",
    "CategoryBreakdown",
    "UncertaintyMethod",
    "ConfidenceLevel",
    # --- Verification Preparation Workflow ---
    "VerificationPreparationWorkflow",
    "VerificationInput",
    "VerificationResult",
    "AuditTrailEntry",
    "ProvenanceHashRecord",
    "CompletenessRequirement",
    "VerificationPackageSection",
    "MethodologyDescription",
    "VerificationLevel",
    "CompletenessStatus",
    "HashVerificationStatus",
    # --- Disclosure Generation Workflow ---
    "DisclosureGenerationWorkflow",
    "DisclosureInput",
    "DisclosureResult",
    "FrameworkRequirement",
    "FrameworkTemplate",
    "ComplianceScore",
    "GapAnalysis",
    "FrameworkOutput",
    "DisclosureFramework",
    "OutputFormat",
    "ComplianceLevel",
    # --- Full Inventory Workflow ---
    "FullInventoryWorkflow",
    "FullInventoryInput",
    "FullInventoryResult",
    "OrganizationStructure",
    "BaseYearConfig",
    "TrendResult",
    "TrendDirection",
    # --- Type Aliases ---
    "BoundaryInput",
    "BoundaryOutput",
    "DataInput",
    "DataOutput",
    "S1Input",
    "S1Output",
    "S2Input",
    "S2Output",
    "ConsolidationInput",
    "ConsolidationOutput",
    "VerifInput",
    "VerifOutput",
    "DiscInput",
    "DiscOutput",
    "FullInput",
    "FullOutput",
    # --- Helpers ---
    "get_loaded_workflows",
]
